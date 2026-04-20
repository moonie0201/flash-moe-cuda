/*
 * expert_ops.cu
 * Fused 2-bit expert forward for MoE single-token inference.
 *
 * Replaces the Python per-expert loop with 3 CUDA kernels:
 *   1. gate_up_gemv_2bit  — dequant + GEMV, output gate_up [N, gu_rows]
 *   2. silu_gate          — in-place SiLU + gating, output h [N, dn_cols]
 *   3. down_gemv_acc_2bit — dequant + GEMV + weighted atomicAdd to output [dn_rows]
 *
 * Grid/block strategy:
 *   Kernels 1 & 3: grid=(rows, N), block=THREADS
 *     Each CTA handles one (expert, output_row) pair.
 *     Threads split the inner-product dimension; warp+block reduce.
 *   Kernel 2: grid=ceil(N*dn_cols/256), block=256 — elementwise SiLU.
 *
 * Memory layout of one expert's raw bytes (3,538,944 bytes total):
 *   [0                  ]  gate_up_codes  [gu_rows * gu_cols / 4]   uint8
 *   [GU_CODES_BYTES      ]  gate_up_scales [gu_rows * n_groups_gu]   float16
 *   [GU_CODES+GU_SCALES  ]  down_codes     [dn_rows * dn_cols / 4]   uint8
 *   [above + DN_CODES    ]  down_scales    [dn_rows * n_groups_dn]   float16
 *
 * Model-specific constants (Qwen3.5-397B):
 *   gu_rows=2048  gu_cols=4096  dn_rows=4096  dn_cols=1024  group_size=64
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>

#define WARP_SIZE 32

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: gate_up GEMV with fused 2-bit dequant
// ─────────────────────────────────────────────────────────────────────────────
template <int THREADS, int GU_COLS>
__global__ void gate_up_gemv_2bit(
    const uint8_t* __restrict__ raw,       // [N, expert_size]
    const __nv_bfloat16* __restrict__ x,   // [GU_COLS]
    float* __restrict__ gate_up,            // [N, gu_rows]
    int N, int gu_rows,
    int group_size, int n_groups_gu,
    long long gu_codes_off,                // byte offset of gate_up codes
    long long gu_scales_off,               // byte offset of gate_up scales
    long long expert_size
) {
    int row = blockIdx.x;
    int exp = blockIdx.y;
    if (row >= gu_rows || exp >= N) return;

    int tid = threadIdx.x;

    const uint8_t* codes  = raw + (long long)exp * expert_size + gu_codes_off;
    const __half*  scales = (const __half*)(raw + (long long)exp * expert_size + gu_scales_off);

    float acc = 0.0f;
    const long long row_base = (long long)row * GU_COLS;

    // Each thread handles 4 cols per iteration (one packed byte).
    // col4 = tid*4 ensures consecutive threads read consecutive bytes → coalesced.
    #pragma unroll 4
    for (int col4 = tid * 4; col4 < GU_COLS; col4 += THREADS * 4) {
        int group = col4 / group_size;
        float scale = __half2float(scales[(long long)row * n_groups_gu + group]);

        uint8_t packed = codes[(row_base + col4) / 4];

        // FMA: (code - 1.5) * scale * x_col
        acc = fmaf(((packed      ) & 0x3) - 1.5f, scale * __bfloat162float(x[col4 + 0]), acc);
        acc = fmaf(((packed >> 2) & 0x3) - 1.5f, scale * __bfloat162float(x[col4 + 1]), acc);
        acc = fmaf(((packed >> 4) & 0x3) - 1.5f, scale * __bfloat162float(x[col4 + 2]), acc);
        acc = fmaf(((packed >> 6) & 0x3) - 1.5f, scale * __bfloat162float(x[col4 + 3]), acc);
    }

    // Warp-level reduction
    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);

    // Cross-warp reduction via shared memory
    constexpr int N_WARPS = THREADS / WARP_SIZE;
    __shared__ float smem[N_WARPS];
    if (tid % WARP_SIZE == 0)
        smem[tid / WARP_SIZE] = acc;
    __syncthreads();

    if (tid < N_WARPS) {
        acc = smem[tid];
        #pragma unroll
        for (int off = N_WARPS / 2; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, off);
        if (tid == 0)
            gate_up[(long long)exp * gu_rows + row] = acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: SiLU + gating in-place
// gate_up[exp, 0..dn_cols-1] = SiLU(gate) * up
// where gate = gate_up[exp, 0..dn_cols-1], up = gate_up[exp, dn_cols..gu_rows-1]
// ─────────────────────────────────────────────────────────────────────────────
__global__ void silu_gate_inplace(
    float* __restrict__ gate_up,  // [N, gu_rows]
    int N, int dn_cols, int gu_rows
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n   = idx / dn_cols;
    int col = idx % dn_cols;
    if (n >= N) return;

    float g = gate_up[(long long)n * gu_rows + col];
    float u = gate_up[(long long)n * gu_rows + col + dn_cols];
    // SiLU(g) = g * sigmoid(g) = g / (1 + exp(-g))
    gate_up[(long long)n * gu_rows + col] = g * __frcp_rn(1.0f + __expf(-g)) * u;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: down GEMV with fused 2-bit dequant + weighted atomic accumulate
// ─────────────────────────────────────────────────────────────────────────────
template <int THREADS, int DN_COLS>
__global__ void down_gemv_acc_2bit(
    const uint8_t* __restrict__ raw,    // [N, expert_size]
    const float* __restrict__ h,         // [N, h_row_stride] (h at cols 0..DN_COLS-1)
    float* __restrict__ output,          // [dn_rows]  (atomic accumulate)
    const float* __restrict__ weights,   // [N]  routing weights
    int N, int dn_rows,
    int group_size, int n_groups_dn,
    long long dn_codes_off,
    long long dn_scales_off,
    long long expert_size,
    int h_row_stride                     // stride between experts in h (= gu_rows)
) {
    int row = blockIdx.x;
    int exp = blockIdx.y;
    if (row >= dn_rows || exp >= N) return;

    int tid = threadIdx.x;

    const uint8_t* codes  = raw + (long long)exp * expert_size + dn_codes_off;
    const __half*  scales = (const __half*)(raw + (long long)exp * expert_size + dn_scales_off);
    // h is stored at the start of each expert's row; stride is gu_rows (not dn_cols)
    const float*   h_exp  = h + (long long)exp * h_row_stride;
    float w = weights[exp];

    float acc = 0.0f;
    const long long row_base = (long long)row * DN_COLS;

    #pragma unroll 4
    for (int col4 = tid * 4; col4 < DN_COLS; col4 += THREADS * 4) {
        int group = col4 / group_size;
        float scale = __half2float(scales[(long long)row * n_groups_dn + group]);

        uint8_t packed = codes[(row_base + col4) / 4];

        acc = fmaf(((packed      ) & 0x3) - 1.5f, scale * h_exp[col4 + 0], acc);
        acc = fmaf(((packed >> 2) & 0x3) - 1.5f, scale * h_exp[col4 + 1], acc);
        acc = fmaf(((packed >> 4) & 0x3) - 1.5f, scale * h_exp[col4 + 2], acc);
        acc = fmaf(((packed >> 6) & 0x3) - 1.5f, scale * h_exp[col4 + 3], acc);
    }

    // Warp reduction
    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);

    constexpr int N_WARPS = THREADS / WARP_SIZE;
    __shared__ float smem[N_WARPS];
    if (tid % WARP_SIZE == 0)
        smem[tid / WARP_SIZE] = acc;
    __syncthreads();

    if (tid < N_WARPS) {
        acc = smem[tid];
        #pragma unroll
        for (int off = N_WARPS / 2; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, off);
        if (tid == 0)
            atomicAdd(&output[row], w * acc);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host entry point
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor expert_batch_forward_2bit(
    torch::Tensor raw_batch,        // [N, expert_size] uint8  (already on GPU)
    torch::Tensor x,                // [gu_cols] bfloat16      (single token)
    torch::Tensor routing_weights,  // [N] float32
    int64_t gu_rows, int64_t gu_cols,
    int64_t dn_rows, int64_t dn_cols,
    int64_t group_size
) {
    const int N = (int)raw_batch.size(0);
    auto device = raw_batch.device();
    // Ensure kernels launch on the same device as the input tensors.
    // Without this, dual-GPU setups (layers on cuda:1) would launch on cuda:0
    // and produce illegal cross-device memory accesses.
    const c10::cuda::CUDAGuard device_guard(device);

    const int n_groups_gu = (int)(gu_cols / group_size);
    const int n_groups_dn = (int)(dn_cols / group_size);

    const long long gu_codes_bytes  = (long long)gu_rows * gu_cols / 4;
    const long long gu_scales_bytes = (long long)gu_rows * n_groups_gu * 2;
    const long long dn_codes_bytes  = (long long)dn_rows * dn_cols / 4;
    const long long expert_size     = gu_codes_bytes + gu_scales_bytes + dn_codes_bytes
                                      + (long long)dn_rows * n_groups_dn * 2;

    const long long gu_codes_off  = 0;
    const long long gu_scales_off = gu_codes_bytes;
    const long long dn_codes_off  = gu_codes_bytes + gu_scales_bytes;
    const long long dn_scales_off = dn_codes_off + dn_codes_bytes;

    // Gate_up intermediate buffer [N, gu_rows] float32
    auto gate_up = torch::zeros({N, (int)gu_rows},
                                 torch::TensorOptions().dtype(torch::kFloat32).device(device));

    const uint8_t*          raw_ptr     = raw_batch.data_ptr<uint8_t>();
    const __nv_bfloat16*    x_ptr       = (const __nv_bfloat16*)x.data_ptr();
    float*                  gate_up_ptr = gate_up.data_ptr<float>();
    const float*            wt_ptr      = routing_weights.data_ptr<float>();

    // --- Kernel 1: gate_up GEMV ---
    // Block: 128 threads (4 warps). Grid: (gu_rows, N).
    // Each CTA: one (row, expert) dot-product over gu_cols=4096.
    // Threads stride by 128*4=512 cols → 4096/512 = 8 iterations of 4 cols each.
    constexpr int K1_THREADS = 128;
    dim3 grid1((int)gu_rows, N);
    gate_up_gemv_2bit<K1_THREADS, 4096><<<grid1, K1_THREADS>>>(
        raw_ptr, x_ptr, gate_up_ptr,
        N, (int)gu_rows, (int)group_size, n_groups_gu,
        gu_codes_off, gu_scales_off, expert_size
    );

    // --- Kernel 2: SiLU + gating ---
    {
        int total = N * (int)dn_cols;
        silu_gate_inplace<<<(total + 255) / 256, 256>>>(
            gate_up_ptr, N, (int)dn_cols, (int)gu_rows
        );
    }

    // --- Kernel 3: down GEMV + weighted accumulate ---
    auto output = torch::zeros({(int)dn_rows},
                                torch::TensorOptions().dtype(torch::kFloat32).device(device));
    float* out_ptr = output.data_ptr<float>();

    // Block: 64 threads (2 warps). Grid: (dn_rows, N).
    // Each CTA: one (row, expert) dot-product over dn_cols=1024.
    // 64 threads × 4 cols × (1024/256) = 4 iterations → covers all 1024.
    constexpr int K3_THREADS = 64;
    dim3 grid3((int)dn_rows, N);
    down_gemv_acc_2bit<K3_THREADS, 1024><<<grid3, K3_THREADS>>>(
        raw_ptr, gate_up_ptr, out_ptr, wt_ptr,
        N, (int)dn_rows, (int)group_size, n_groups_dn,
        dn_codes_off, dn_scales_off, expert_size,
        (int)gu_rows   // h_row_stride: gate_up allocated as [N, gu_rows]
    );

    return output.to(torch::kBFloat16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("expert_batch_forward_2bit", &expert_batch_forward_2bit,
          "Fused 2-bit dequant + expert MoE forward (CUDA), single-token path.\n"
          "Args: raw_batch[N,E] uint8, x[gu_cols] bf16, routing_weights[N] f32,\n"
          "      gu_rows, gu_cols, dn_rows, dn_cols, group_size\n"
          "Returns: output[dn_rows] bf16");
}
