# Experiment 4: SES 통합 파이프라인

## 목적

Hierarchical Expert Decomposition + Confidence-Aware Prefetch + Speculative Computation 세 가지 pillar를 통합한 SES 파이프라인 구현 및 검증.

---

## 의존성

- Experiment 1 (SVD 분해): Base/Residual 분리 데이터 생성 완료
- Experiment 3 (CPU 추론): 기본 추론 엔진 동작 확인
- CUDA Toolkit: GPU 예측기 구현

---

## Step 1: Expert 재패킹 (Base/Residual 분리)

### 1.1 repack_experts_ses.py

Experiment 1에서 결정된 최적 rank로 전체 expert를 Base + Residual로 분리 패킹:

```python
# repack_experts_ses.py

import numpy as np
import os
import time
from pathlib import Path

EXPERT_SIZE = 7077888
NUM_EXPERTS = 512
NUM_LAYERS = 60
OPTIMAL_RANK = 64  # Experiment 1에서 결정

# Base expert layout (rank-64 기준, 추정)
# gate_proj base: U[1024,64] FP16 + S[64] FP32 + Vt[64,4096] FP16
# = 131072 + 256 + 524288 = 655,616 bytes
# up_proj base: 동일 = 655,616 bytes
# down_proj base: U[4096,64] FP16 + S[64] FP32 + Vt[64,1024] FP16
# = 524288 + 256 + 131072 = 655,616 bytes
# Total base per expert ≈ 1,966,848 bytes (~1.9MB)

BASE_EXPERT_SIZE = None   # Experiment 1 결과로 결정
RES_EXPERT_SIZE = None    # EXPERT_SIZE - BASE_EXPERT_SIZE (양자화 후)

def decompose_and_pack_layer(layer_idx, packed_dir, output_dir, rank=OPTIMAL_RANK):
    """한 레이어의 512 experts를 Base + Residual로 분리"""
    
    src_path = f"{packed_dir}/layer_{layer_idx:02d}.bin"
    base_path = f"{output_dir}/layer_{layer_idx:02d}_base.bin"
    res_path = f"{output_dir}/layer_{layer_idx:02d}_residual.bin"
    
    t0 = time.monotonic()
    
    # Pre-allocate output files
    base_fd = os.open(base_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    res_fd = os.open(res_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(base_fd, NUM_EXPERTS * BASE_EXPERT_SIZE)
    os.ftruncate(res_fd, NUM_EXPERTS * RES_EXPERT_SIZE)
    
    with open(src_path, 'rb') as f:
        for expert_idx in range(NUM_EXPERTS):
            # Read original expert
            f.seek(expert_idx * EXPERT_SIZE)
            expert_data = f.read(EXPERT_SIZE)
            
            # Dequantize each projection
            for proj_name, offset, size, shape in PROJECTIONS:
                weight_bytes = expert_data[offset:offset+size]
                matrix = dequant_4bit(weight_bytes, ...)
                
                # SVD decompose
                U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
                
                # Base: rank-r factors (stored as U_r, S_r, Vt_r)
                base_U = U[:, :rank].astype(np.float16)
                base_S = S[:rank].astype(np.float32)
                base_Vt = Vt[:rank, :].astype(np.float16)
                
                # Residual: W - U_r @ diag(S_r) @ Vt_r, re-quantized to 4-bit
                base_approx = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
                residual = matrix - base_approx
                res_packed = quantize_4bit(residual)
                
                # Write to output files
                write_base_component(base_fd, expert_idx, proj_name, 
                                    base_U, base_S, base_Vt)
                write_res_component(res_fd, expert_idx, proj_name, res_packed)
    
    os.close(base_fd)
    os.close(res_fd)
    
    elapsed = time.monotonic() - t0
    print(f"Layer {layer_idx:02d}: {elapsed:.1f}s")
```

### 1.2 출력 파일 레이아웃

```
packed_experts_ses/
├── layout.json           # Base/Residual 크기, rank, format 정보
├── layer_00_base.bin     # 512 × BASE_EXPERT_SIZE (~1GB)
├── layer_00_residual.bin # 512 × RES_EXPERT_SIZE (~2.6GB)
├── layer_01_base.bin
├── layer_01_residual.bin
...
├── layer_59_base.bin
└── layer_59_residual.bin

Total Base: ~60GB (원본 209GB의 ~29%)
Total Residual: ~156GB
Total: ~216GB (원본 + 약간의 오버헤드)
```

---

## Step 2: GPU 예측기 + Confidence 분석 (CUDA)

### 2.1 predict.cu

```cuda
// gate projection 커널 (dequant_matvec_4bit의 CUDA 버전)
// 512 output × 4096 input, 4-bit quantized
__global__ void gate_matvec_4bit(
    const uint32_t* __restrict__ weight,    // [512, 512] (packed)
    const uint16_t* __restrict__ scales,    // [512, 64]
    const uint16_t* __restrict__ biases,    // [512, 64]
    const float*    __restrict__ input,     // [4096]
    float*          __restrict__ output,    // [512]
    int out_dim, int in_dim, int group_size
) {
    // Block: 256 threads = 8 warps
    // Each warp: 1 output row
    // ROWS_PER_BLOCK = 8
    
    __shared__ float x_shared[4096];  // input vector cache
    
    // Cooperative load of input vector
    int tid = threadIdx.x;
    for (int i = tid; i < in_dim; i += blockDim.x)
        x_shared[i] = input[i];
    __syncthreads();
    
    int row = blockIdx.x * 8 + (tid / 32);  // 8 rows per block
    if (row >= out_dim) return;
    
    int lane = tid % 32;  // warp lane
    int packed_cols = in_dim / 8;
    int num_groups = in_dim / group_size;
    
    float acc = 0.0f;
    
    for (int p = lane; p < packed_cols; p += 32) {
        uint32_t packed = weight[row * packed_cols + p];
        int base_col = p * 8;
        int group = base_col / group_size;
        
        // BF16 → FP32
        uint16_t s_raw = scales[row * num_groups + group];
        uint16_t b_raw = biases[row * num_groups + group];
        float scale = __uint_as_float(((uint32_t)s_raw) << 16);
        float bias = __uint_as_float(((uint32_t)b_raw) << 16);
        
        // FMA optimization
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            float nibble = (float)((packed >> (n * 4)) & 0xF);
            float x = x_shared[base_col + n];
            acc = fmaf(nibble, scale * x, bias * x);
            // Simplified: acc += (nibble * scale + bias) * x;
        }
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    
    if (lane == 0)
        output[row] = acc;
}
```

### 2.2 Confidence 분석 (CPU, GPU 결과 수신 후)

```c
// predictor.c

typedef enum {
    CONF_HIGH,    // top-4가 지배적
    CONF_MEDIUM,  // 중간
    CONF_LOW      // 평탄한 분포
} ConfidenceLevel;

typedef struct {
    int expert_ids[32];      // 예측된 expert (최대 32개)
    float expert_scores[32];
    int num_full;             // Full prefetch 대상 수
    int num_base;             // Base prefetch 대상 수
    ConfidenceLevel confidence;
} PredictionResult;

void analyze_confidence(const float* gate_scores, int num_experts, 
                       PredictionResult* result) {
    // Softmax
    float max_score = gate_scores[0];
    for (int i = 1; i < num_experts; i++)
        if (gate_scores[i] > max_score) max_score = gate_scores[i];
    
    float sum = 0.0f;
    float probs[512];
    for (int i = 0; i < num_experts; i++) {
        probs[i] = expf(gate_scores[i] - max_score);
        sum += probs[i];
    }
    for (int i = 0; i < num_experts; i++)
        probs[i] /= sum;
    
    // Top-K mass
    // (partial sort로 top-32 추출)
    int indices[512];
    for (int i = 0; i < num_experts; i++) indices[i] = i;
    // ... partial sort top-32 ...
    
    float top4_mass = 0.0f;
    for (int k = 0; k < 4; k++)
        top4_mass += probs[indices[k]];
    
    // Entropy
    float entropy = 0.0f;
    for (int i = 0; i < num_experts; i++)
        if (probs[i] > 1e-10f)
            entropy -= probs[i] * logf(probs[i]);
    float norm_entropy = entropy / logf((float)num_experts);
    
    // Classify
    if (top4_mass > 0.8f && norm_entropy < 0.3f) {
        result->confidence = CONF_HIGH;
        result->num_full = 4;
        result->num_base = 0;
    } else if (top4_mass > 0.5f) {
        result->confidence = CONF_MEDIUM;
        result->num_full = 4;
        result->num_base = 12;
    } else {
        result->confidence = CONF_LOW;
        result->num_full = 4;
        result->num_base = 28;
    }
    
    // Fill expert_ids
    int total = result->num_full + result->num_base;
    for (int k = 0; k < total; k++) {
        result->expert_ids[k] = indices[k];
        result->expert_scores[k] = probs[indices[k]];
    }
}
```

---

## Step 3: mmap + madvise 계층적 프리페치

### 3.1 ses_io.c

```c
// ses_io.c

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

typedef struct {
    void* base_mmap[60];     // Base 파일 mmap
    void* res_mmap[60];      // Residual 파일 mmap
    size_t base_expert_size;
    size_t res_expert_size;
    size_t base_file_size[60];
    size_t res_file_size[60];
} SES_IO;

void ses_io_init(SES_IO* io, const char* ses_dir) {
    for (int l = 0; l < 60; l++) {
        char path[256];
        
        // Base 파일 mmap
        snprintf(path, sizeof(path), "%s/layer_%02d_base.bin", ses_dir, l);
        int fd = open(path, O_RDONLY);
        struct stat st;
        fstat(fd, &st);
        io->base_file_size[l] = st.st_size;
        io->base_mmap[l] = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        madvise(io->base_mmap[l], st.st_size, MADV_RANDOM);
        close(fd);
        
        // Residual 파일 mmap
        snprintf(path, sizeof(path), "%s/layer_%02d_residual.bin", ses_dir, l);
        fd = open(path, O_RDONLY);
        fstat(fd, &st);
        io->res_file_size[l] = st.st_size;
        io->res_mmap[l] = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        madvise(io->res_mmap[l], st.st_size, MADV_RANDOM);
        close(fd);
    }
}

void ses_prefetch(SES_IO* io, int layer, const PredictionResult* pred) {
    // Full prefetch 대상: Base + Residual 모두 MADV_WILLNEED
    for (int k = 0; k < pred->num_full; k++) {
        int eid = pred->expert_ids[k];
        // Base
        madvise((char*)io->base_mmap[layer] + eid * io->base_expert_size,
                io->base_expert_size, MADV_WILLNEED);
        // Residual
        madvise((char*)io->res_mmap[layer] + eid * io->res_expert_size,
                io->res_expert_size, MADV_WILLNEED);
    }
    
    // Base-only prefetch 대상: Base만 MADV_WILLNEED
    for (int k = pred->num_full; k < pred->num_full + pred->num_base; k++) {
        int eid = pred->expert_ids[k];
        madvise((char*)io->base_mmap[layer] + eid * io->base_expert_size,
                io->base_expert_size, MADV_WILLNEED);
    }
}

// 페이지가 메모리에 있는지 확인 (mincore)
int ses_check_resident(SES_IO* io, int layer, int expert_id, int check_residual) {
    unsigned char vec[256];  // enough for 7MB / 4KB pages
    
    void* base_addr = (char*)io->base_mmap[layer] + expert_id * io->base_expert_size;
    int base_pages = (io->base_expert_size + 4095) / 4096;
    mincore(base_addr, io->base_expert_size, vec);
    
    int base_resident = 1;
    for (int i = 0; i < base_pages; i++)
        if (!(vec[i] & 1)) { base_resident = 0; break; }
    
    if (!check_residual || !base_resident)
        return base_resident;
    
    void* res_addr = (char*)io->res_mmap[layer] + expert_id * io->res_expert_size;
    int res_pages = (io->res_expert_size + 4095) / 4096;
    mincore(res_addr, io->res_expert_size, vec);
    
    int res_resident = 1;
    for (int i = 0; i < res_pages; i++)
        if (!(vec[i] & 1)) { res_resident = 0; break; }
    
    return base_resident + res_resident;  // 0=none, 1=base only, 2=full
}
```

---

## Step 4: Speculative Expert Forward Pass

### 4.1 ses_expert.c

```c
// ses_expert.c

typedef struct {
    // Base: SVD factors (U_r, S_r, Vt_r) per projection
    float* gate_U;   // [1024, rank] (or FP16)
    float* gate_S;   // [rank]
    float* gate_Vt;  // [rank, 4096]
    float* up_U;     // [1024, rank]
    float* up_S;     // [rank]
    float* up_Vt;    // [rank, 4096]
    float* down_U;   // [4096, rank]
    float* down_S;   // [rank]
    float* down_Vt;  // [rank, 1024]
} ExpertBase;

void ses_expert_forward_base(const void* base_data, int rank,
                              const float* x, float* output, int dim) {
    ExpertBase base;
    parse_base_data(base_data, rank, &base);
    
    // gate(x) = U @ diag(S) @ Vt @ x
    // 효율적 순서: (1) tmp = Vt @ x [rank], (2) tmp *= S [rank], (3) out = U @ tmp [1024]
    float tmp[256];  // max rank
    
    // gate projection (base)
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rank, dim,
                1.0f, base.gate_Vt, dim, x, 1, 0.0f, tmp, 1);
    for (int i = 0; i < rank; i++) tmp[i] *= base.gate_S[i];
    float gate_out[1024];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 1024, rank,
                1.0f, base.gate_U, rank, tmp, 1, 0.0f, gate_out, 1);
    
    // up projection (base) — 동일 패턴
    float up_out[1024];
    // ... 동일 ...
    
    // SwiGLU
    float act[1024];
    cpu_swiglu(gate_out, up_out, act, 1024);
    
    // down projection (base)
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rank, 1024,
                1.0f, base.down_Vt, 1024, act, 1, 0.0f, tmp, 1);
    for (int i = 0; i < rank; i++) tmp[i] *= base.down_S[i];
    cblas_sgemv(CblasRowMajor, CblasNoTrans, dim, rank,
                1.0f, base.down_U, rank, tmp, 1, 0.0f, output, 1);
}

void ses_expert_forward_full(const void* base_data, const void* res_data,
                              int rank, const float* x, float* output, int dim) {
    // Base forward
    float base_output[4096];
    ses_expert_forward_base(base_data, rank, x, base_output, dim);
    
    // Residual forward (4-bit dequant matvec, 기존 방식)
    float res_output[4096];
    cpu_expert_forward_residual(res_data, x, res_output, dim);
    
    // Combine: output = base + residual
    for (int i = 0; i < dim; i++)
        output[i] = base_output[i] + res_output[i];
}
```

---

## Step 5: 통합 SES 파이프라인

```c
// ses_pipeline.c — 메인 추론 루프

typedef struct {
    int layer;
    int expert_id;
    int quality;  // 0=miss, 1=base_only, 2=full
} ExpertLoadStat;

void ses_inference(const char* prompt, int max_tokens) {
    // 초기화
    SES_IO io;
    ses_io_init(&io, "packed_experts_ses");
    
    Predictor predictor;
    predictor_init(&predictor, "gate_weights/", /*gpu_device=*/0);
    
    // 통계
    int total_experts = 0, full_hits = 0, base_hits = 0, misses = 0;
    
    for (int token = 0; token < max_tokens; token++) {
        for (int layer = 0; layer < 60; layer++) {
            
            // === GPU PREDICTION (async, 다음 레이어) ===
            PredictionResult pred;
            if (layer < 59) {
                predict_experts_gpu(&predictor, layer + 1, hidden, &pred);
                ses_prefetch(&io, layer + 1, &pred);
            }
            
            // === CPU: ATTENTION ===
            cpu_attention(layer, hidden, ...);
            
            // === CPU: ACTUAL ROUTING ===
            float gate_scores[512];
            cpu_gate_projection(layer, hidden, gate_scores);
            int actual_ids[4];
            float actual_weights[4];
            cpu_softmax_topk(gate_scores, 512, actual_ids, actual_weights, 4);
            
            // === SPECULATIVE EXPERT FORWARD ===
            float expert_outputs[4][4096];
            
            for (int k = 0; k < 4; k++) {
                int eid = actual_ids[k];
                int residency = ses_check_resident(&io, layer, eid, /*check_res=*/1);
                
                void* base = (char*)io.base_mmap[layer] + eid * io.base_expert_size;
                void* res = (char*)io.res_mmap[layer] + eid * io.res_expert_size;
                
                if (residency == 2) {
                    // Full hit: Base + Residual 모두 메모리에 있음
                    ses_expert_forward_full(base, res, OPTIMAL_RANK, 
                                           hidden, expert_outputs[k], 4096);
                    full_hits++;
                } else if (residency >= 1) {
                    // Base hit: Base만 메모리에 있음 → speculative compute
                    ses_expert_forward_base(base, OPTIMAL_RANK,
                                           hidden, expert_outputs[k], 4096);
                    base_hits++;
                    // Note: quality는 degraded하지만 stall 없음
                } else {
                    // Complete miss: 동기 로드 (fallback)
                    // Base가 작으므로 Full보다 빠른 fallback 가능
                    ses_expert_forward_base(base, OPTIMAL_RANK,
                                           hidden, expert_outputs[k], 4096);
                    misses++;
                }
                total_experts++;
            }
            
            // === COMBINE + RESIDUAL + NORM ===
            cpu_moe_combine(expert_outputs, actual_weights, hidden, 4, 4096);
            cpu_rms_norm(hidden, ...);
        }
        
        // 토큰 샘플링
        int next_token = sample(hidden, ...);
        
        // 통계 출력
        if (token % 10 == 0) {
            printf("Token %d: full=%.1f%% base=%.1f%% miss=%.1f%% | %.2f tok/s\n",
                   token,
                   100.0 * full_hits / total_experts,
                   100.0 * base_hits / total_experts,
                   100.0 * misses / total_experts,
                   tokens_per_second);
        }
    }
}
```

---

## 측정 지표

| 지표 | 설명 | 목표 |
|---|---|---|
| Full hit rate | Residual까지 프리페치 성공 | > 50% |
| Base hit rate | Base만 프리페치 성공 | > 30% |
| Miss rate | 아무것도 없음 | < 20% |
| Effective quality | base_only 레이어의 출력 품질 | cosine sim > 0.95 |
| tok/s | 초당 토큰 생성 속도 | baseline 대비 +30% |
| I/O bytes/token | 토큰당 SSD 읽기량 | baseline 대비 -40% |

## 성공 기준

1. SES가 no-prefetch 대비 **+30% tok/s** 이상
2. Base-only degradation이 최종 generation quality에 **perplexity +5% 이내** 영향
3. Confidence-aware가 fixed top-4 대비 **effective coverage +15%** 이상

## 예상 소요 시간

- Expert 재패킹: 2-3일 (I/O bound, 216GB 처리)
- GPU 예측기: 3-5일
- mmap + prefetch: 2-3일
- Speculative forward: 2-3일
- 통합 + 디버깅: 3-5일
- 총: 2-3주
