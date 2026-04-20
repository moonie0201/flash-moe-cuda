# Experiment 3: CPU Inference Engine Porting

## 목적

flash-moe의 Metal/Objective-C 추론 엔진을 Linux CPU 전용으로 포팅. SES 실험의 기반 시스템.

---

## 포팅 범위

### 원본 파일
- `metal_infer/infer.m` (7,151줄) — 전체 추론 엔진
- `metal_infer/shaders.metal` (1,296줄) — Metal GPU 커널 26개
- `metal_infer/chat.m` — 대화형 TUI
- `metal_infer/tokenizer.h` — C BPE 토크나이저 (449줄, 재사용)
- `metal_infer/Makefile` — 빌드 시스템

### 새로 생성할 파일
- `infer_cpu.c` — CPU 추론 엔진 (infer.m 포팅)
- `cpu_kernels.c` / `cpu_kernels.h` — CPU 커널 구현
- `Makefile.linux` — Linux 빌드

---

## Step 1: CPU 커널 구현

### 1.1 Dequant MatVec (핵심, Metal `dequant_matvec_4bit_v3` 대응)

```c
// cpu_kernels.c

#include <omp.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>  // AVX2

// BF16 → FP32 변환
static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t f32_bits = ((uint32_t)bf16) << 16;
    float result;
    memcpy(&result, &f32_bits, 4);
    return result;
}

// 4-bit dequantized matrix-vector multiply
// weight: [out_dim × in_dim/8] uint32 (8 nibbles packed)
// scales: [out_dim × num_groups] bf16
// biases: [out_dim × num_groups] bf16
// input:  [in_dim] float
// output: [out_dim] float
void cpu_dequant_matvec_4bit(
    const uint32_t* weight,
    const uint16_t* scales,
    const uint16_t* biases,
    const float* input,
    float* output,
    int out_dim,
    int in_dim,
    int group_size  // typically 64
) {
    int num_groups = in_dim / group_size;
    int packed_cols = in_dim / 8;
    
    #pragma omp parallel for schedule(dynamic, 8)
    for (int row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        const uint32_t* w_row = weight + row * packed_cols;
        
        for (int g = 0; g < num_groups; g++) {
            float scale = bf16_to_f32(scales[row * num_groups + g]);
            float bias = bf16_to_f32(biases[row * num_groups + g]);
            
            int col_start = g * group_size;
            int pack_start = col_start / 8;
            int pack_end = (col_start + group_size) / 8;
            
            for (int p = pack_start; p < pack_end; p++) {
                uint32_t packed = w_row[p];
                int base_col = p * 8;
                
                // FMA optimization (same as Metal kernel):
                // (nibble * scale + bias) * x = fma(nibble, scale*x, bias*x)
                for (int n = 0; n < 8; n++) {
                    float nibble = (float)((packed >> (n * 4)) & 0xF);
                    float x = input[base_col + n];
                    acc = fmaf(nibble, scale * x, bias * x + acc) - acc + acc;
                    // Simplified: acc += (nibble * scale + bias) * x;
                    acc += (nibble * scale + bias) * x;
                }
            }
        }
        output[row] = acc;
    }
}
```

### 1.2 RMS Norm

```c
void cpu_rms_norm(const float* input, const float* weight, float* output,
                  int dim, float eps) {
    // Pass 1: sum of squares
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++)
        sum_sq += input[i] * input[i];
    
    float rms = 1.0f / sqrtf(sum_sq / dim + eps);
    
    // Pass 2: apply
    for (int i = 0; i < dim; i++)
        output[i] = input[i] * rms * weight[i];
}
```

### 1.3 SwiGLU

```c
void cpu_swiglu(const float* gate, const float* up, float* output, int dim) {
    for (int i = 0; i < dim; i++) {
        float sigmoid = 1.0f / (1.0f + expf(-gate[i]));
        output[i] = gate[i] * sigmoid * up[i];  // silu(gate) * up
    }
}
```

### 1.4 Attention (Full Attention, 15 layers)

```c
void cpu_attention_forward(
    const float* Q, const float* K, const float* V,
    float* output,
    int num_heads, int head_dim, int seq_len
) {
    // Q @ K^T → scores
    // softmax(scores / sqrt(head_dim)) → weights  
    // weights @ V → output
    // OpenBLAS cblas_sgemv 활용
}
```

### 1.5 GatedDeltaNet (Linear Attention, 45 layers)

```c
// 기존 infer.m의 CPU BLAS 경로를 그대로 포팅
// cblas_sscal, cblas_sgemv, cblas_sger 사용
void cpu_gated_delta_net_step(
    float* state,       // [num_heads, v_dim, k_dim] = [64, 128, 128]
    const float* q,     // [num_heads, k_dim]
    const float* k,     // [num_heads, k_dim]
    const float* v,     // [num_heads, v_dim]
    const float* beta,  // [num_heads]
    const float* gate,  // [num_heads]
    float* output,      // [num_heads, v_dim]
    int num_heads, int k_dim, int v_dim
) {
    for (int h = 0; h < num_heads; h++) {
        float* S = state + h * v_dim * k_dim;
        
        // S *= gate[h]
        cblas_sscal(v_dim * k_dim, gate[h], S, 1);
        
        // output[h] = S @ q[h]
        cblas_sgemv(CblasRowMajor, CblasNoTrans, v_dim, k_dim,
                    1.0f, S, k_dim, q + h * k_dim, 1,
                    0.0f, output + h * v_dim, 1);
        
        // S += beta[h] * outer(v[h], k[h])
        cblas_sger(CblasRowMajor, v_dim, k_dim,
                   beta[h], v + h * v_dim, 1, k + h * k_dim, 1,
                   S, k_dim);
    }
}
```

### 1.6 MoE Combine + Residual

```c
void cpu_moe_combine_residual(
    const float* expert_outputs[],  // [K] arrays of [dim]
    const float* weights,           // [K] routing weights
    const float* shared_output,     // [dim]
    float* residual,                // [dim] input residual, updated in-place
    int K, int dim
) {
    // weighted sum of experts
    float gate_val = 0.0f;  // sigmoid gate for shared expert
    
    for (int i = 0; i < dim; i++) {
        float expert_sum = 0.0f;
        for (int k = 0; k < K; k++)
            expert_sum += weights[k] * expert_outputs[k][i];
        
        // sigmoid gate on shared expert
        float sig = 1.0f / (1.0f + expf(-shared_output[i]));
        
        residual[i] += expert_sum + sig * shared_output[i];
    }
}
```

---

## Step 2: 호스트 코드 포팅 (infer.m → infer_cpu.c)

### 주요 변경 사항

| 원본 (Metal/ObjC) | 포팅 (C/Linux) |
|---|---|
| `#import <Metal/Metal.h>` | 제거 |
| `@autoreleasepool {}` | 제거 |
| `MTLDevice`, `MTLCommandQueue` | 제거 |
| `MTLBuffer` | `aligned_alloc(64, size)` |
| `[buffer contents]` | 직접 포인터 |
| `MTLComputePipelineState` | 직접 CPU 함수 호출 |
| `newCommandBuffer` / `commit` / `wait` | 직접 호출 (동기) |
| `dispatch_group_async` | `pthread` pool |
| `#import <Accelerate/Accelerate.h>` | `#include <cblas.h>` (OpenBLAS) |
| `-framework Metal -framework Accelerate` | `-lopenblas -lm -lpthread -fopenmp` |

### 코드 구조

```c
// infer_cpu.c

#include "cpu_kernels.h"
#include "tokenizer.h"   // 그대로 재사용
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/time.h>

// Weight loading (mmap)
typedef struct {
    void* data;
    size_t size;
} MappedFile;

MappedFile map_file(const char* path) {
    int fd = open(path, O_RDONLY);
    struct stat st;
    fstat(fd, &st);
    MappedFile mf = {
        .data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0),
        .size = st.st_size
    };
    close(fd);
    return mf;
}

// Expert loading (pread, 기존 방식)
typedef struct {
    int fd;
    size_t expert_size;
} LayerFile;

void load_expert(LayerFile* lf, int expert_idx, void* dst) {
    pread(lf->fd, dst, lf->expert_size, (off_t)expert_idx * lf->expert_size);
}

// Per-layer forward pass
void layer_forward(int layer_idx, float* hidden, ...) {
    // 1. Attention input projections (CPU matvec)
    cpu_dequant_matvec_4bit(q_proj_w, ..., hidden, q_out, ...);
    cpu_dequant_matvec_4bit(k_proj_w, ..., hidden, k_out, ...);
    cpu_dequant_matvec_4bit(v_proj_w, ..., hidden, v_out, ...);
    
    // 2. Attention (delta-net or full)
    if (is_linear_attention[layer_idx]) {
        cpu_gated_delta_net_step(...);
    } else {
        cpu_attention_forward(...);
    }
    
    // 3. O projection + residual + norm
    cpu_dequant_matvec_4bit(o_proj_w, ..., attn_out, o_out, ...);
    cpu_residual_add(hidden, o_out, dim);
    cpu_rms_norm(hidden, norm_w, hidden, dim, 1e-6);
    
    // 4. Gate + routing
    cpu_dequant_matvec_4bit(gate_w, ..., hidden, gate_scores, ...);
    softmax(gate_scores, 512);
    topk(gate_scores, 512, expert_ids, expert_weights, K);
    
    // 5. Load experts (pread)
    for (int k = 0; k < K; k++)
        load_expert(&layer_files[layer_idx], expert_ids[k], expert_bufs[k]);
    
    // 6. Expert forward
    for (int k = 0; k < K; k++) {
        cpu_expert_forward(expert_bufs[k], hidden, expert_outs[k]);
    }
    
    // 7. Shared expert forward
    cpu_shared_expert_forward(...);
    
    // 8. Combine + residual + norm
    cpu_moe_combine_residual(expert_outs, expert_weights, shared_out, hidden, K, dim);
    cpu_rms_norm(hidden, norm_w2, hidden, dim, 1e-6);
}
```

---

## Step 3: 빌드 + 검증

### Makefile.linux

```makefile
CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall
LDFLAGS = -lopenblas -lm -lpthread

all: infer_cpu

infer_cpu: infer_cpu.c cpu_kernels.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f infer_cpu
```

### 검증 방법

1. **단위 테스트**: 각 CPU 커널의 출력을 numpy 참조와 비교
   ```bash
   python verify_kernels.py  # numpy로 동일 연산 수행, C 출력과 비교
   ```

2. **통합 테스트**: 전체 추론 출력 확인
   ```bash
   ./infer_cpu --prompt "Hello" --tokens 20 --timing
   ```

3. **성능 측정**:
   ```bash
   ./infer_cpu --prompt "Explain quantum computing" --tokens 50 --timing
   # 레이어별 시간 출력
   ```

---

## 예상 성능

| 구간 | 예상 시간/layer | 비고 |
|---|---|---|
| Attention projections (CPU) | ~5-10ms | OpenMP 병렬, matvec [4096→4096] × 4 |
| Delta-net recurrence | ~0.3ms | OpenBLAS, 기존과 동일 |
| Expert I/O | ~2-8ms | pread, page cache hit 의존 |
| Expert forward (CPU) | ~5-10ms | 4개 expert × matvec [1024→4096] × 3 |
| Combine + norm | ~0.1ms | |
| **Total/layer** | **~15-30ms** | |
| **tok/s** | **~0.5-1.1** | 60 layers × 15-30ms = 0.9-1.8s/token |

CPU 전용이므로 Mac의 4.36 tok/s (GPU+SSD)보다 느리지만, SES의 기반 시스템으로 충분.

## 예상 소요 시간

- CPU 커널 구현: 3-5일
- 호스트 코드 포팅: 5-7일  
- 빌드 + 디버깅: 2-3일
- 총: 2-3주
