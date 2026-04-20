# Experiment 1: Expert SVD Decomposition Quality

## 목적

각 expert의 weight matrix를 SVD로 Base(low-rank) + Residual로 분해했을 때, Base만으로 충분한 출력 품질을 유지하는지 검증.

**이 실험이 전체 논문의 성패를 결정**. rank-64에서 cosine similarity > 0.95가 나오지 않으면 SES의 근본 전제가 무너짐.

---

## 배경

Expert 내부 구조 (repack_experts.py 기준):
```
gate_proj: weight[1024, 512] U32 + scales[1024, 64] BF16 + biases[1024, 64] BF16
up_proj:   weight[1024, 512] U32 + scales[1024, 64] BF16 + biases[1024, 64] BF16
down_proj: weight[4096, 128] U32 + scales[4096, 16] BF16 + biases[4096, 16] BF16
```

Note: weight shape에서 두 번째 차원은 packed dimension (4-bit, 8 nibbles per uint32).
실제 matrix shape:
- gate_proj: [1024, 4096] (1024 output, 4096 input = hidden_dim)
- up_proj: [1024, 4096]
- down_proj: [4096, 1024] (4096 output = hidden_dim, 1024 input = intermediate)

## 실험 절차

### Step 1: Expert 데이터 추출 + Dequantize

```python
# ses_decompose.py

import numpy as np
import struct
from pathlib import Path

# 4-bit affine quantization format (MLX style)
# weight: uint32 packed (8 nibbles per word)
# scales: bf16 per group
# biases: bf16 per group  
# group_size = 64

def bf16_to_f32(raw_bytes):
    """BF16 → FP32 변환"""
    values = np.frombuffer(raw_bytes, dtype=np.uint16)
    # BF16: sign(1) + exp(8) + mantissa(7), FP32의 상위 16bit와 동일
    f32_bits = values.astype(np.uint32) << 16
    return np.frombuffer(f32_bits.tobytes(), dtype=np.float32)

def dequant_4bit(weight_bytes, scale_bytes, bias_bytes, out_dim, in_dim, group_size=64):
    """4-bit packed weight → FP32 matrix"""
    weight_u32 = np.frombuffer(weight_bytes, dtype=np.uint32)  # [out_dim, in_dim/8]
    scales = bf16_to_f32(scale_bytes).reshape(out_dim, in_dim // group_size)
    biases = bf16_to_f32(bias_bytes).reshape(out_dim, in_dim // group_size)
    
    result = np.zeros((out_dim, in_dim), dtype=np.float32)
    
    for row in range(out_dim):
        for col_pack in range(in_dim // 8):
            packed = weight_u32[row * (in_dim // 8) + col_pack]
            for n in range(8):
                nibble = (packed >> (n * 4)) & 0xF
                col = col_pack * 8 + n
                group = col // group_size
                result[row, col] = nibble * scales[row, group] + biases[row, group]
    
    return result

def load_expert(layer_file, expert_idx, expert_size=7077888):
    """packed_experts/layer_XX.bin에서 expert 하나 로드"""
    offset = expert_idx * expert_size
    with open(layer_file, 'rb') as f:
        f.seek(offset)
        data = f.read(expert_size)
    
    # Component offsets (from repack_experts.py)
    return {
        'gate_proj': {
            'weight': data[0:2097152],
            'scales': data[2097152:2228224],
            'biases': data[2228224:2359296],
            'shape': (1024, 4096),
        },
        'up_proj': {
            'weight': data[2359296:4456448],
            'scales': data[4456448:4587520],
            'biases': data[4587520:4718592],
            'shape': (1024, 4096),
        },
        'down_proj': {
            'weight': data[4718592:6815744],
            'scales': data[6815744:6946816],
            'biases': data[6946816:7077888],
            'shape': (4096, 1024),
        },
    }
```

### Step 2: SVD 분해 + Rank별 품질 측정

```python
def svd_decompose(matrix, rank):
    """Matrix를 rank-r approximation으로 분해"""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    # Base: rank-r approximation
    base = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
    
    # Residual
    residual = matrix - base
    
    # Metrics
    total_energy = np.sum(S**2)
    captured_energy = np.sum(S[:rank]**2)
    energy_ratio = captured_energy / total_energy
    
    recon_error = np.linalg.norm(residual, 'fro') / np.linalg.norm(matrix, 'fro')
    
    return base, residual, {
        'rank': rank,
        'energy_ratio': energy_ratio,
        'relative_error': recon_error,
        'singular_values': S[:min(128, len(S))].tolist(),
        'base_size_bytes': rank * (matrix.shape[0] + matrix.shape[1]) * 4,  # FP32
        'original_size_bytes': matrix.shape[0] * matrix.shape[1] * 4,
    }

def analyze_expert(layer_file, expert_idx, ranks=[16, 32, 64, 128, 256]):
    """하나의 expert에 대해 전체 분석"""
    expert = load_expert(layer_file, expert_idx)
    results = {}
    
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        proj = expert[proj_name]
        matrix = dequant_4bit(
            proj['weight'], proj['scales'], proj['biases'],
            proj['shape'][0], proj['shape'][1]
        )
        
        results[proj_name] = {}
        for rank in ranks:
            base, residual, metrics = svd_decompose(matrix, rank)
            results[proj_name][rank] = metrics
            print(f"  {proj_name} rank-{rank}: "
                  f"energy={metrics['energy_ratio']:.4f}, "
                  f"error={metrics['relative_error']:.4f}, "
                  f"compression={metrics['base_size_bytes']/metrics['original_size_bytes']:.2%}")
    
    return results
```

### Step 3: Forward Pass 품질 비교

```python
def expert_forward(gate_w, up_w, down_w, x):
    """Standard MoE expert forward: down(SwiGLU(gate(x), up(x)))"""
    gate_out = gate_w @ x           # [1024]
    up_out = up_w @ x               # [1024]
    
    # SwiGLU: gate * sigmoid(gate) * up
    swiglu = gate_out * (1 / (1 + np.exp(-gate_out))) * up_out  # [1024]
    
    output = down_w @ swiglu        # [4096]
    return output

def compare_base_vs_full(layer_file, expert_idx, ranks=[16, 32, 64, 128]):
    """Base-only vs Full expert의 output 비교"""
    expert = load_expert(layer_file, expert_idx)
    
    # Dequantize full matrices
    gate_full = dequant_4bit(...)
    up_full = dequant_4bit(...)
    down_full = dequant_4bit(...)
    
    # Random input (simulate hidden state)
    np.random.seed(42)
    x = np.random.randn(4096).astype(np.float32)
    x = x / np.linalg.norm(x)  # normalize
    
    # Full forward
    out_full = expert_forward(gate_full, up_full, down_full, x)
    
    for rank in ranks:
        # SVD decompose each projection
        gate_base, _, _ = svd_decompose(gate_full, rank)
        up_base, _, _ = svd_decompose(up_full, rank)
        down_base, _, _ = svd_decompose(down_full, rank)
        
        # Base-only forward
        out_base = expert_forward(gate_base, up_base, down_base, x)
        
        # Quality metrics
        cosine_sim = np.dot(out_full, out_base) / (np.linalg.norm(out_full) * np.linalg.norm(out_base))
        l2_error = np.linalg.norm(out_full - out_base) / np.linalg.norm(out_full)
        
        print(f"  rank-{rank}: cosine_sim={cosine_sim:.6f}, relative_l2={l2_error:.6f}")
    
    return results
```

### Step 4: 다수 Expert/Layer 통계

```python
def full_analysis(packed_dir, num_layers=60, experts_per_layer=10, ranks=[32, 64, 128]):
    """여러 layer, 여러 expert에 대해 통계 수집"""
    all_results = []
    
    for layer_idx in range(num_layers):
        layer_file = f"{packed_dir}/layer_{layer_idx:02d}.bin"
        
        # 각 layer에서 10개 expert 랜덤 샘플링
        expert_indices = np.random.choice(512, experts_per_layer, replace=False)
        
        for expert_idx in expert_indices:
            result = compare_base_vs_full(layer_file, expert_idx, ranks)
            all_results.append({
                'layer': layer_idx,
                'expert': expert_idx,
                **result
            })
    
    # 통계 요약
    for rank in ranks:
        sims = [r[rank]['cosine_sim'] for r in all_results]
        print(f"\nRank-{rank} across {len(all_results)} experts:")
        print(f"  Cosine similarity: mean={np.mean(sims):.4f}, "
              f"std={np.std(sims):.4f}, min={np.min(sims):.4f}")
```

---

## 측정 지표

| 지표 | 목표 | 의미 |
|---|---|---|
| Energy ratio (rank-64) | > 0.95 | SVD 상위 64 singular value가 전체 에너지의 95% 이상 |
| Reconstruction error | < 0.10 | Frobenius norm 기준 10% 미만 |
| Output cosine similarity | > 0.95 | Base-only forward와 Full forward의 출력 방향 일치 |
| Output relative L2 error | < 0.15 | 출력 크기 차이 15% 미만 |
| Base size ratio | < 0.25 | Base가 원본의 25% 이하 |

## 성공/실패 기준

| 결과 | 판단 | 다음 단계 |
|---|---|---|
| cosine_sim > 0.95 at rank-64 | **성공** | Phase 2 진행 |
| cosine_sim 0.85-0.95 at rank-64 | **부분 성공** | rank 올리거나, critical layer만 Full 사용 |
| cosine_sim < 0.85 at rank-64 | **실패** | SVD 대신 다른 분해 방법 탐색 (NMF, structured pruning 등) |

## 대안 분해 방법 (SVD 실패 시)

1. **행 기반 분할**: importance score 기반으로 상위 N 행 = Base, 나머지 = Residual
2. **Column-wise partitioning**: input dimension을 분할
3. **Structured pruning**: magnitude 기반으로 작은 weight를 0으로
4. **Mixed-precision Base**: Base를 8-bit, Residual을 4-bit로 다르게 양자화

## 예상 소요 시간

- Step 1 (데이터 추출): 1-2시간 (I/O bound)
- Step 2 (SVD): expert당 ~30초, 10개 × 60 layer = ~5시간
- Step 3 (Forward 비교): ~1시간
- Step 4 (통계): ~6시간
- 총: 1-2일
