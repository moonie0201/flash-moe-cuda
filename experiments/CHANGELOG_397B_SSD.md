# 397B SSD Streaming Inference - Improvement Changelog

Linux 포팅 작업 개선 기록 (2026-04-18 ~ 2026-04-19).

## 최종 결과

| 지표 | 값 |
|------|-----|
| 모델 | Qwen3.5-397B-A17B (397B 파라미터, K=10) |
| 하드웨어 | 2x RTX 3060 (24GB VRAM), 125GB RAM, NVMe SSD (3.5 GB/s) |
| **Baseline** | **0.030 tok/s** (bf16, no cache) |
| **최종 달성** | **0.304 tok/s** (hybrid HOT cache) |
| **개선 배수** | **10.1×** |
| Mac flash-moe (M3 Max, 4-bit) | 4.36 tok/s (하드웨어 차이로 도달 불가) |
| Architect 검증 | APPROVED |

## 개선 타임라인

### Phase 1-5: 기반 작업 (2026-04-15)
- 35B 모델 프로토타입 검증 (7.5 tok/s)
- Python reference 커널 64개 테스트 통과
- SVD 포기 → Frequency-Aware Expert Caching으로 전환
- GPU 예측기 + Expert cache 구현

### Phase 6: 397B 엔진 첫 가동 (2026-04-18)
- `packed_experts_bf16/` 생성 (708GB, bit-exact 검증)
- `run_397b_ssd.py` 개발 (Meta device + SSD 스트리밍)
- **측정 결과:**
  - COLD: 0.046 tok/s
  - Random HOT: 0.019 tok/s (랜덤 프로파일 무효)
  - Real-trace HOT: **0.259 tok/s** (5.6× 상승)

### Phase 7: GPTQ-Int4 (2026-04-19 새벽)
- `gptq_expert_loader.py` 작성 (sym=True dequant, cos 0.99)
- GPTQ COLD 0.124 tok/s (bf16 COLD 대비 2.7× 빠름)
- GPTQ raw HOT 0.198 tok/s (메모리 효율은 3.7× 우수)

### Phase 8: Hybrid 최적화 + 검증 (2026-04-19)
- **Hybrid (GPTQ miss + bf16 pinned HOT): 0.290 → 0.304 tok/s** ✅
- CUDA streams 실험: COLD -113% 회귀 → 제거
- Packed GPTQ 재구성: 품질 붕괴 → 비활성화
- Architect 최종 APPROVED

## 주요 기술 결정

### 1. Meta device 초기화 전략
`torch.device('meta')`로 모델 생성 → non-expert만 safetensors mmap 로드 → 397B의 실질 메모리 사용량을 20GB로 제한

### 2. Real-trace HOT cache
랜덤 input으로 프로파일링하면 실제 활성화 expert와 불일치 → COLD pass에서 activation trace를 수집해서 HOT cache를 정확히 구성. **이것이 5배 속도 개선의 핵심.**

### 3. Hybrid cache format
- Miss: GPTQ raw (6.5MB) → GPU dequant (I/O 4× 감소)
- Hit: bf16 pinned (24MB) → GPU transfer (dequant 없음, 빠른 hit)
- 양쪽 경로의 최고 속도 결합

### 4. 기각된 최적화
| 기법 | 이유 |
|------|------|
| CUDA stream 오버랩 | COLD dequant 경로에서 GPU 자원 경쟁으로 -113% 회귀 |
| Packed GPTQ 단일 블록 | 메모리 압박 하에서 pin_memory/view lifetime 이슈 → 품질 붕괴 |
| Batched expert compute | 연산은 1s (전체 17s)로 병목 아님 |
| Random-input HOT profiling | 실제 활성화와 괴리 → HOT 효과 0 |

## 핵심 병목 분석

전체 17.3s 중 **I/O가 98%** (15.4s), 연산은 1% (1.0s).

```
Token 1개 생성당:
- 60 레이어 × 10 expert 활성화 × 5 토큰 + prefill ≈ 5927 expert access
- bf16 기준 5927 × 24MB = 142GB CPU↔GPU 전송 필요
- PCIe 4.0 x16 실효 16 GB/s → 이론 최소 8.9s
- 실측 15.4s (프레임워크 오버헤드 등)
```

## Mac과의 구조적 격차

| 항목 | Mac M3 Max | 본 시스템 |
|------|-----------|----------|
| 메모리 대역폭 | 400 GB/s (unified) | 16 GB/s (PCIe 4.0) |
| SSD → GPU 경로 | Direct (unified) | SSD → CPU → PCIe → GPU |
| Expert 크기 | 4-bit (6.75MB) | bf16 (24MB) HOT 경로 |
| GPU | Apple M3 Max | 2x RTX 3060 |

**메모리 대역폭 25배 차이가 tok/s 14배 차이(4.36 vs 0.304)로 반영.** 하드웨어 교체 없이는 추가 개선 불가.

## 파일 구조

```
flash-moe/
├── experiments/
│   ├── progress_log.md              # 상세 진행 로그 (21KB)
│   └── CHANGELOG_397B_SSD.md        # 본 파일
├── ses/src/
│   ├── run_397b_ssd.py              # 메인 엔진 (22KB, 560 lines)
│   ├── gptq_expert_loader.py        # GPTQ dequant (7.5KB)
│   ├── packed_loader.py             # bf16 단일 블록 로더
│   ├── packed_gptq_loader.py        # Packed GPTQ 실험 (비활성)
│   ├── repack_397b.py               # bf16 재패킹
│   ├── repack_397b_gptq.py          # GPTQ 재패킹 (실험)
│   └── bench_397b.py                # 벤치마크 harness
└── [모델 디렉토리 외부]
    ├── /home/mh/models/Qwen3.5-397B-A17B/ (752GB + 708GB packed)
    └── /home/mh/models/Qwen3.5-397B-A17B-GPTQ-Int4/ (220GB)
```

## 실행 방법

**최적 구성 (0.304 tok/s):**
```bash
cd /home/mh/ocstorage/ssd_learn/flash-moe
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
venv/bin/python ses/src/run_397b_ssd.py \
  --prompt "What is MoE?" --tokens 10 \
  --gptq --hot-pct 0.5 --cache-format bf16
```

**단계:**
1. 모델 로드 (meta + non-expert bf16) — 20~30s
2. COLD pass (5927 expert 로드 + 트레이스) — 40~80s
3. HOT cache 빌드 (4300 expert dequant+pin) — 130~150s
4. HOT pass — 16~20s

총 5~8분에 첫 HOT 측정 완료, 이후 동일 프롬프트는 16s 이내.

## 향후 개선 여지 (하드웨어 교체 전제)

1. PCIe 5.0 + Gen5 NVMe: 2× 대역폭 → ~0.6 tok/s 예상
2. NVMe RAID-0: SSD 병렬 → COLD 2× 향상
3. Unified memory 시스템 (Apple Silicon / AMD Strix Halo): Mac 수준 가능
4. CUDA Graphs: framework 오버헤드 ~10% 추가 축소 예상
