# 397B 개선 실측 결과

## Baseline
- 397B bf16, no cache, no prediction
- **0.030 tok/s**, cache hit rate 0%
- GPU: 19.6GB (9.2 + 10.4)

## Improvement 1: HOT Cache (bf16)
- 397B bf16, HOT 5% = 1500 experts in CPU pinned memory
- Cache hit rate: 6.1% (random input profiling)
- **0.029 tok/s** (개선 미미)

### 분석
- bf16 expert 크기 25MB → HOT 5%만 해도 37.7GB pinned
- 더 큰 hot_pct는 RAM 125GB 초과로 OOM
- Random input profiling → 실제 inference activation과 괴리
- **결론**: bf16 자체가 병목. 4-bit로 전환 필요.

## Improvement 2: 4-bit GPTQ (예정)
- GPTQ-Int4 모델 사용
- Expert 크기 6.5MB → HOT 30% = 55GB pinned (RAM OK)
- 예상: 0.1-0.5 tok/s
