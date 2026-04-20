# Experiment 5: End-to-End Benchmark

## 목적

SES 파이프라인의 최종 성능과 품질을 체계적으로 측정하고, 기존 방법들과 비교하여 논문용 실험 결과 생성.

---

## 의존성

- Experiment 1-4 모두 완료
- 전체 SES 파이프라인 동작 확인

---

## Baseline 시스템 (비교 대상)

| System | 설명 |
|---|---|
| **No-Prefetch** | 동기식 pread, 예측 없음 (flash-moe 원본의 Linux 포팅) |
| **Temporal** | 이전 토큰의 routing 재사용 (25% hit rate) |
| **Fate-style** | Cross-layer gate prediction, top-4 full prefetch |
| **SES (ours)** | Hierarchical + Confidence-aware + Speculative |
| **SES-ablation-1** | Hierarchical only (no confidence, no speculative) |
| **SES-ablation-2** | Hierarchical + Confidence (no speculative) |
| **Oracle** | 미래 routing을 완벽히 알고 있는 이상적 시스템 (upper bound) |

---

## Benchmark 1: Throughput (tok/s)

### 설정
```bash
# 각 시스템에 대해 동일 프롬프트로 측정
PROMPTS=(
    "Explain the theory of general relativity in simple terms."
    "Write a Python function to sort a list using quicksort."
    "What are the economic implications of artificial intelligence?"
)
TOKENS=100

for system in no_prefetch temporal fate ses ses_ablation1 ses_ablation2 oracle; do
    for prompt in "${PROMPTS[@]}"; do
        ./infer_${system} --prompt "${prompt}" --tokens ${TOKENS} --timing \
            2>&1 | tee results/${system}_$(echo ${prompt} | md5sum | cut -c1-8).log
    done
done
```

### 측정 항목
```
Per-layer breakdown:
  - attention_ms: attention 연산 시간
  - routing_ms: gate projection + softmax + topK
  - prediction_ms: GPU 예측 시간 (SES만)
  - io_ms: expert 로딩 시간
  - expert_ms: expert forward pass 시간
  - combine_ms: combine + residual + norm

Per-token:
  - total_ms: 전체 토큰 생성 시간
  - tok_per_sec: 초당 토큰

Per-session:
  - avg_tok_s: 평균 tok/s
  - first_token_ms: 첫 토큰 지연시간 (TTFT)
  - p50/p95/p99 token latency
```

### 기대 결과 표

| System | tok/s | I/O ms/layer | Expert ms/layer | Total ms/layer |
|---|---|---|---|---|
| No-Prefetch | ~0.5-0.8 | ~8 | ~8 | ~25 |
| Temporal | ~0.6-0.9 | ~6 | ~8 | ~22 |
| Fate-style | ~0.7-1.0 | ~4 | ~8 | ~19 |
| **SES** | **~0.9-1.3** | **~2** | **~6** | **~14** |
| Oracle | ~1.2-1.5 | ~0.5 | ~8 | ~12 |

---

## Benchmark 2: Generation Quality

### 2.1 Perplexity (WikiText-2)

```python
# eval_perplexity.py

def evaluate_perplexity(system, dataset="wikitext-2"):
    """시스템별 perplexity 측정"""
    total_nll = 0.0
    total_tokens = 0
    
    for text in load_dataset(dataset):
        tokens = tokenize(text)
        for i in range(1, len(tokens)):
            # 각 시스템으로 next-token logits 계산
            logits = run_inference(system, tokens[:i])
            nll = -log_softmax(logits)[tokens[i]]
            total_nll += nll
            total_tokens += 1
    
    perplexity = exp(total_nll / total_tokens)
    return perplexity
```

### 2.2 MMLU (다지선다 정확도)

```bash
# 5-shot MMLU evaluation
python eval_mmlu.py --system ses --shots 5 --subjects all
python eval_mmlu.py --system no_prefetch --shots 5 --subjects all
```

### 2.3 HumanEval (코드 생성)

```bash
python eval_humaneval.py --system ses --temperature 0.0
python eval_humaneval.py --system no_prefetch --temperature 0.0
```

### 기대 결과 표

| System | PPL (WikiText-2) | MMLU (5-shot) | HumanEval (pass@1) |
|---|---|---|---|
| No-Prefetch (Full) | baseline | baseline | baseline |
| SES (mixed) | +2-5% | -0.5-1.0% | -1-2% |
| SES Base-only | +5-15% | -2-4% | -3-5% |

**핵심 claim**: SES의 품질 저하가 속도 향상 대비 미미함.

---

## Benchmark 3: Ablation Study

### 3.1 각 Pillar의 개별 기여도

| Config | Hierarchical | Confidence | Speculative | 예상 tok/s |
|---|---|---|---|---|
| Baseline (no prefetch) | - | - | - | 0.6 |
| + Hierarchical only | O | - | - | 0.8 |
| + Confidence | O | O | - | 0.9 |
| + Speculative (full SES) | O | O | O | 1.1 |

### 3.2 Rank 민감도

| Rank | Base Size | Quality (cos sim) | tok/s |
|---|---|---|---|
| 16 | ~0.5MB | ~0.85 | 1.3 |
| 32 | ~1.0MB | ~0.92 | 1.2 |
| **64** | **~1.9MB** | **~0.96** | **1.1** |
| 128 | ~3.5MB | ~0.99 | 0.9 |

### 3.3 Confidence 임계값 민감도

| HIGH threshold | MEDIUM range | Coverage | Quality | tok/s |
|---|---|---|---|---|
| top4_mass > 0.9 | 0.6-0.9 | LOW coverage | HIGH quality | 0.9 |
| top4_mass > 0.8 | 0.5-0.8 | MED coverage | MED quality | 1.1 |
| top4_mass > 0.7 | 0.4-0.7 | HIGH coverage | LOW quality | 1.0 |

---

## Benchmark 4: I/O 분석

### 4.1 SSD 읽기 프로파일링

```bash
# blktrace로 I/O 패턴 추적
sudo blktrace -d /dev/nvme0n1 -o trace &
./infer_ses --prompt "Hello" --tokens 50
sudo kill %1
blkparse trace.blktrace.* | python analyze_io.py
```

### 4.2 Page Cache 분석

```bash
# vmtouch로 page cache 상태 확인
vmtouch packed_experts_ses/layer_00_base.bin
vmtouch packed_experts_ses/layer_00_residual.bin

# perf로 page fault 카운트
perf stat -e page-faults,minor-faults,major-faults \
    ./infer_ses --prompt "Hello" --tokens 50
```

### 4.3 MADV_WILLNEED 효과 측정

```c
// ses 내부에서 측정
struct timeval before_madvise, after_expert_access;
gettimeofday(&before_madvise, NULL);
madvise(addr, size, MADV_WILLNEED);
// ... attention 연산 (시간 벌기) ...
gettimeofday(&after_expert_access, NULL);  // expert 실제 접근 시
// prefetch_lead_time = after - before
// actual_access_time = 시간 측정
```

---

## Benchmark 5: Speculative Quality Spectrum 시각화

논문의 핵심 figure — hit/miss 이진 vs SES 연속 스펙트럼:

```python
# plot_quality_spectrum.py

import matplotlib.pyplot as plt
import numpy as np

def plot_spectrum(ses_stats):
    """각 expert access의 quality level 분포 시각화"""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 상단: 기존 시스템 (binary)
    # hit=green, miss=red
    ax1 = axes[0]
    ax1.set_title("Existing: Binary Hit/Miss")
    # ... bar chart ...
    
    # 하단: SES (continuous spectrum)
    # full=green, base=yellow, miss=red
    ax2 = axes[1]
    ax2.set_title("SES: Continuous Quality Spectrum")
    colors = {'full': 'green', 'base': 'gold', 'miss': 'red'}
    # ... stacked bar chart per layer ...
    
    plt.savefig('results/quality_spectrum.png', dpi=150)
```

---

## 논문용 Figure 목록

1. **Fig 1**: Architecture diagram — GPU prediction → SSD prefetch → CPU speculative compute
2. **Fig 2**: Quality spectrum — Binary (기존) vs Continuous (SES)
3. **Fig 3**: SVD rank vs quality (cosine similarity curves per projection)
4. **Fig 4**: Confidence distribution across layers (heatmap)
5. **Fig 5**: Throughput comparison (bar chart, all systems)
6. **Fig 6**: Quality vs Speed tradeoff curve (Pareto frontier)
7. **Fig 7**: Ablation study (stacked bar, each pillar's contribution)
8. **Fig 8**: Per-layer I/O analysis (timeline, prefetch vs actual access)

---

## 논문용 Table 목록

1. **Table 1**: System comparison (tok/s, quality, I/O)
2. **Table 2**: SVD rank sensitivity (rank, size, quality, speed)
3. **Table 3**: Confidence threshold sensitivity
4. **Table 4**: Ablation results
5. **Table 5**: Generation quality (PPL, MMLU, HumanEval)

---

## 실험 환경 기록

```bash
# 실험 시작 전 환경 스냅샷
nvidia-smi > results/env_gpu.txt
cat /proc/cpuinfo | head -30 > results/env_cpu.txt
free -h > results/env_memory.txt
lsblk > results/env_storage.txt
nvme smart-log /dev/nvme0n1 > results/env_nvme.txt
uname -a > results/env_kernel.txt
nvcc --version > results/env_cuda.txt

# NVMe 벤치마크
fio --name=expert_read \
    --rw=randread \
    --bs=7M \
    --numjobs=4 \
    --iodepth=4 \
    --runtime=30 \
    --time_based \
    --filename=packed_experts/layer_00.bin \
    > results/fio_baseline.txt

fio --name=base_read \
    --rw=randread \
    --bs=2M \
    --numjobs=4 \
    --iodepth=16 \
    --runtime=30 \
    --time_based \
    --filename=packed_experts_ses/layer_00_base.bin \
    > results/fio_base.txt
```

## 예상 소요 시간

- Throughput 벤치마크: 1-2일
- Quality 벤치마크: 2-3일 (PPL, MMLU, HumanEval)
- Ablation: 1-2일
- I/O 분석: 1일
- 시각화 + 논문 figure: 2-3일
- 총: 1-2주
