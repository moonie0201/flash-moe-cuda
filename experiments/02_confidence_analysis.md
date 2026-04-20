# Experiment 2: Confidence-Aware Activation Landscape Analysis

## 목적

MoE routing의 activation landscape를 분석하여, 예측 confidence에 따라 프리페치 전략을 동적으로 전환하는 것이 고정 top-K 대비 효과적인지 검증.

---

## 배경

기존 연구 (Fate, PreScope 등)는 항상 top-K (K=4) expert만 프리페치. SES는 예측 불확실성을 정량화하여:
- **확실할 때**: top-4 Full 프리페치 (기존과 동일)
- **불확실할 때**: top-16~32 Base 프리페치 (넓게, 가볍게)

이를 위해 activation landscape의 특성을 먼저 파악해야 함.

---

## 실험 절차

### Step 1: Gate Score 수집

실제 추론 중 각 layer의 gate score 분포를 기록:

```python
# collect_gate_scores.py

def collect_scores(model_weights, packed_experts, prompts, num_tokens=100):
    """추론 중 모든 layer의 gate score를 수집"""
    all_scores = []  # [token][layer] = scores[512]
    
    for token_idx in range(num_tokens):
        token_scores = []
        for layer in range(60):
            # gate projection: hidden[4096] → scores[512]
            scores = gate_weight[layer] @ hidden  # [512]
            scores = softmax(scores)
            token_scores.append(scores)
            
            # 정상 추론 계속...
            actual_experts = topk(scores, K=4)
            # ... expert forward pass ...
        
        all_scores.append(token_scores)
    
    return all_scores  # shape: [num_tokens, 60, 512]
```

### Step 2: Landscape 특성 분석

```python
def analyze_landscape(all_scores):
    """각 layer/token의 activation landscape 특성 분석"""
    
    for layer in range(60):
        for token_scores in all_scores:
            scores = token_scores[layer]  # [512]
            
            # Metric 1: Entropy (평탄도)
            entropy = -np.sum(scores * np.log(scores + 1e-10))
            max_entropy = np.log(512)  # uniform distribution
            normalized_entropy = entropy / max_entropy
            
            # Metric 2: Top-K mass (집중도)
            sorted_scores = np.sort(scores)[::-1]
            top4_mass = np.sum(sorted_scores[:4])
            top8_mass = np.sum(sorted_scores[:8])
            top16_mass = np.sum(sorted_scores[:16])
            
            # Metric 3: Gini coefficient (불균등도)
            n = len(scores)
            sorted_asc = np.sort(scores)
            gini = (2 * np.sum((np.arange(1, n+1)) * sorted_asc) / (n * np.sum(sorted_asc))) - (n+1)/n
            
            # Metric 4: Effective number of experts
            effective_n = np.exp(entropy)  # perplexity of distribution
```

### Step 3: Confidence Level 분류 + 최적 전략 매핑

```python
def classify_confidence(scores):
    """Gate scores로부터 confidence level 결정"""
    sorted_scores = np.sort(scores)[::-1]
    top4_mass = np.sum(sorted_scores[:4])
    entropy = -np.sum(scores * np.log(scores + 1e-10))
    normalized_entropy = entropy / np.log(len(scores))
    
    if top4_mass > 0.8 and normalized_entropy < 0.3:
        return 'HIGH'      # top-4가 지배적 → Full prefetch
    elif top4_mass > 0.5:
        return 'MEDIUM'    # 중간 → top-4 Full + top-12 Base
    else:
        return 'LOW'       # 평탄 → top-4 Full + top-32 Base

def simulate_prefetch_strategies(all_scores):
    """3가지 전략의 effective hit rate 비교"""
    
    strategies = {
        'fixed_top4': {'full': 4, 'base': 0},
        'fixed_top16_base': {'full': 0, 'base': 16},
        'ses_adaptive': None,  # confidence에 따라 동적
    }
    
    for strategy_name, config in strategies.items():
        total_hits_full = 0
        total_hits_base = 0
        total_misses = 0
        total_io_bytes = 0
        
        for token_scores in all_scores:
            for layer in range(60):
                scores = token_scores[layer]
                actual_top4 = np.argsort(scores)[-4:]
                
                if strategy_name == 'ses_adaptive':
                    conf = classify_confidence(scores)
                    if conf == 'HIGH':
                        prefetch_full = set(np.argsort(scores)[-4:])
                        prefetch_base = set()
                        io_bytes = 4 * 7_077_888  # 4 Full experts
                    elif conf == 'MEDIUM':
                        prefetch_full = set(np.argsort(scores)[-4:])
                        prefetch_base = set(np.argsort(scores)[-16:]) - prefetch_full
                        io_bytes = 4 * 7_077_888 + 12 * 1_500_000  # 4 Full + 12 Base
                    else:
                        prefetch_full = set(np.argsort(scores)[-4:])
                        prefetch_base = set(np.argsort(scores)[-32:]) - prefetch_full
                        io_bytes = 4 * 7_077_888 + 28 * 1_500_000  # 4 Full + 28 Base
                else:
                    prefetch_full = set(np.argsort(scores)[-config['full']:]) if config['full'] > 0 else set()
                    prefetch_base = set(np.argsort(scores)[-config['base']:]) if config['base'] > 0 else set()
                    io_bytes = config['full'] * 7_077_888 + config['base'] * 1_500_000
                
                for expert_id in actual_top4:
                    if expert_id in prefetch_full:
                        total_hits_full += 1
                    elif expert_id in prefetch_base:
                        total_hits_base += 1
                    else:
                        total_misses += 1
                
                total_io_bytes += io_bytes
        
        total = total_hits_full + total_hits_base + total_misses
        print(f"\n{strategy_name}:")
        print(f"  Full hits: {total_hits_full/total:.1%}")
        print(f"  Base hits: {total_hits_base/total:.1%}")
        print(f"  Misses:    {total_misses/total:.1%}")
        print(f"  Avg I/O:   {total_io_bytes/len(all_scores)/60/1e6:.1f} MB/layer")
```

### Step 4: Cross-Layer Prediction과 결합

```python
def cross_layer_prediction_with_confidence(all_scores):
    """Fate 방식 cross-layer prediction에 SES confidence를 결합"""
    
    for token_idx, token_scores in enumerate(all_scores):
        for layer in range(1, 60):
            prev_scores = token_scores[layer - 1]
            curr_scores = token_scores[layer]
            
            # Fate: 이전 layer의 top-K로 현재 layer 예측
            predicted = set(np.argsort(prev_scores)[-4:])
            actual = set(np.argsort(curr_scores)[-4:])
            fate_hits = len(predicted & actual)
            
            # SES: confidence에 따라 coverage 확장
            conf = classify_confidence(prev_scores)
            if conf == 'HIGH':
                ses_predicted_full = set(np.argsort(prev_scores)[-4:])
                ses_predicted_base = set()
            elif conf == 'MEDIUM':
                ses_predicted_full = set(np.argsort(prev_scores)[-4:])
                ses_predicted_base = set(np.argsort(prev_scores)[-16:]) - ses_predicted_full
            else:
                ses_predicted_full = set(np.argsort(prev_scores)[-4:])
                ses_predicted_base = set(np.argsort(prev_scores)[-32:]) - ses_predicted_full
            
            ses_hits_full = len(ses_predicted_full & actual)
            ses_hits_base = len(ses_predicted_base & actual)
            ses_total_coverage = ses_hits_full + ses_hits_base
```

---

## 측정 지표

| 지표 | 설명 |
|---|---|
| Normalized entropy | 0=하나만 활성, 1=균등분포. Layer별 분포 파악 |
| Top-K mass | Top-4/8/16이 차지하는 확률 질량 |
| Confidence 분포 | HIGH/MEDIUM/LOW 비율 (layer별) |
| Effective hit rate | 각 전략의 실질 hit rate (Full hit + Base hit) |
| I/O efficiency | hit당 소비한 I/O bytes |

## 핵심 질문

1. Entropy/top-K mass의 layer별 분포는? (shallow vs deep layers 차이?)
2. HIGH/MEDIUM/LOW confidence의 비율은?
3. SES adaptive가 fixed top-4 대비 effective coverage를 얼마나 올리나?
4. 추가 I/O (Base prefetch)의 비용 대비 coverage 이득은?

## 성공 기준

| 조건 | 판단 |
|---|---|
| SES adaptive coverage > fixed top-4 + 15% | **성공** — confidence 분류가 유의미 |
| I/O overhead < 2x | **성공** — Base가 작아서 추가 I/O 비용 합리적 |
| LOW confidence 비율 > 10% | **필요** — adaptive 전략이 작동할 충분한 기회 |

## 예상 소요 시간

- Gate score 수집: 추론 엔진 필요 (Phase 2 이후)
- 분석: ~2-3시간
- 총: Phase 2 완료 후 1일

## 대안 (추론 엔진 없이 사전 분석)

모델 weight가 있으면 gate projection weight만으로 간접 분석 가능:
- 랜덤 hidden state에 gate projection 적용
- Singular value spectrum 분석으로 landscape 특성 추정
- 이는 Step 1의 proxy 실험으로 Phase 1에서 수행 가능
