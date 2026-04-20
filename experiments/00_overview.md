# Speculative Expert Streaming (SES) - 실험 개요

## 논문 제목 (후보)
**"Speculative Expert Streaming: Turning Hit-or-Miss Expert Prefetching into a Continuous Spectrum via Hierarchical Decomposition"**

## 핵심 주장 (Claims)

1. MoE expert를 Base(low-rank) + Residual로 분해하면, Base만으로 85-90% 출력 품질 유지
2. 예측 confidence에 따라 프리페치 전략을 동적 전환하면, 고정 top-K 대비 effective coverage 향상
3. Base 도착 즉시 speculative computation을 시작하면, I/O 대기 시간이 추론 중단 없이 품질 저하로 변환
4. 1+2+3 결합으로 기존 hit/miss 이진 프레임을 연속 스펙트럼으로 전환

## 실험 목록

| # | 실험 | 목적 | 의존성 | 파일 |
|---|---|---|---|---|
| 1 | Expert SVD 분해 품질 | Pillar 1 검증 (핵심) | 모델 weight | `01_svd_decomposition.md` |
| 2 | Confidence 분석 | Pillar 2 검증 | 모델 weight + 추론 | `02_confidence_analysis.md` |
| 3 | CPU 추론 엔진 포팅 | 기반 시스템 | CUDA, OpenBLAS | `03_cpu_inference.md` |
| 4 | SES 통합 파이프라인 | Pillar 3 검증 + 전체 | Phase 1-3 완료 | `04_ses_pipeline.md` |
| 5 | End-to-End 벤치마크 | 논문 실험 | 전체 시스템 | `05_benchmark.md` |

## 실험 순서 (Critical Path)

```
Phase 0: 환경 구축 + 모델 다운로드
    ↓
Phase 1: Experiment 1 (SVD 분해) ← 논문의 성패 결정
    ↓ (rank-64에서 cosine sim > 0.95 확인)
Phase 2: Experiment 3 (CPU 추론 엔진)
    ↓ (정확한 추론 동작 확인)
Phase 3: Experiment 2 + 4 (Confidence + SES 통합)
    ↓
Phase 4: Experiment 5 (벤치마크 + 논문)
```

## 하드웨어 환경

- GPU: 2x NVIDIA RTX 3060 (12GB VRAM, GA104/GA106)
- CPU: (확인 필요)
- RAM: (확인 필요)
- SSD: NVMe (벤치마크 필요)
- OS: Linux 6.17.0 (x86_64)
- CUDA: 드라이버 590.48.01, toolkit 미설치

## 모델

- Qwen3.5-397B-A17B (MoE)
- 60 layers, 512 experts/layer, K=4 activated
- Expert size: 7,077,888 bytes (4-bit quantized)
- Total expert weights: 209GB
- Non-expert weights: 5.5GB

## 기존 연구 참조

| 논문 | 핵심 기여 | SES와의 관계 |
|---|---|---|
| [Fate (2502.12224)](https://arxiv.org/abs/2502.12224) | Cross-layer gate prediction, 97% accuracy | 예측 기법 참조 (SES의 predictor에 적용 가능) |
| [Pre-Attention (2511.10676)](https://arxiv.org/abs/2511.10676) | 같은 레이어 pre-attention prediction, 93-97% | 예측 정확도 향상 기법 |
| [PreScope (2509.23638)](https://arxiv.org/abs/2509.23638) | Learnable predictor + async I/O, +141% throughput | 스케줄링 전략 참조 |
| [KTransformers (SOSP '25)](https://dl.acm.org/doi/10.1145/3731569.3764843) | CPU/GPU hybrid, AMX/AVX-512 | CPU 추론 최적화 참조 |
| [FlashMoE (2601.17063)](https://arxiv.org/abs/2601.17063) | ML-based cache for SSD, +51% hit rate | SSD 캐싱 전략 참조 |
| [MoE-SpeQ (2511.14102)](https://arxiv.org/abs/2511.14102) | Quantized model as predictor, 90.9% | Speculative 개념 참조 |

## Novelty 요약

| 측면 | 기존 전체 | SES |
|---|---|---|
| Expert 취급 | Atomic (all-or-nothing) | Hierarchical (Base + Residual) |
| 예측 결과 | Binary hit/miss | Continuous quality spectrum |
| Miss penalty | Full I/O wait | Graceful degradation |
| Prefetch 전략 | 고정 top-K | Confidence-aware adaptive |
| 연산 시작 | Full expert 도착 후 | Base 도착 즉시 (speculative) |
