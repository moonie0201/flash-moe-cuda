# Experiment Log — 397B MoE on Dual RTX 3060

All experiments on Qwen3.5-397B-A17B, 2-bit quantized, dual RTX 3060 12GB (24GB total VRAM), 96GB DDR4.

## Baseline

| Configuration | tok/s | Notes |
|---|---|---|
| llama.cpp (Q4_K_M, 3 GPU layers) | 1.15 | Best achievable with llama.cpp on this HW |
| SSD only (cold, no cache) | 0.77 | Streaming every expert from SSD per token |
| GPU VRAM cache only (6.8GB) | 1.09 | ~1980 experts cached in VRAM |
| **SES: GPU + CPU + SSD 3-tier** | **1.35** | Final system, +17% vs llama.cpp |

## What Worked

### Float8 Non-Expert Weights
Quantize attention/projection weights (non-expert) to float8.  
**Result**: Freed 7.64GB VRAM (19.6GB → 12.0GB). Enabled larger GPU expert cache.  
**Why it works**: Non-expert weights dominate VRAM; float8 has negligible quality loss for these.

### GPU Hot Cache (6.8GB, ~1980 experts)
Cache the most frequently activated experts in VRAM.  
**Result**: 57% cache hit rate. Single biggest throughput win.

### CPU Pinned Hot Cache (30GB, ~9216 experts)
Cache next tier of frequent experts in pinned CPU RAM; H2D transfer on hit.  
**Result**: 43% hit rate on remaining misses. H2D latency ~0.26ms vs 1.3ms SSD.

### Layer Split Across GPUs
GPU0: layers 0–23. GPU1: layers 24–59.  
**Result**: Allows both GPUs to be utilized; non-expert weights fit in VRAM.

### Custom CUDA Kernel (expert_ops.cu)
Fused 2-bit dequant + batched GEMV + SiLU in one kernel.  
**Result**: Replaces 10 serial expert calls with 1 batched launch. Reduces kernel launch overhead.

### Thread Pool I/O (16 workers)
SSD expert reads issued in parallel via thread pool.  
**Result**: Overlaps SSD latency across the 10 experts per layer.

### Activation Trace (COLD pass)
First pass through model builds expert frequency map → identifies HOT experts to cache.  
**Result**: Enables frequency-aware cache population without manual tuning.

---

## What Failed

### torch.compile
**Result**: +0.75%. GPU compute was never the bottleneck.  
**Why**: Bottleneck is PCIe H2D transfer (13 GB/s), not GPU compute.

### Cross-Layer N-gram Prefetch (per-expert top-K table)
Build a table: for each (layer, expert_id), record which experts tend to activate at layer+1.  
Predict next layer's experts at end of current layer; start H2D during attention.  
N-gram accuracy: 82.8% (in-distribution).  
**Result: -21% throughput (1.062 tok/s).**  
**Root cause**: Table predicted top-4 per expert → ~700 H2D starts per token vs 310 actually needed. PCIe congestion from over-prediction killed performance.

### Cross-Layer N-gram Prefetch (global top-10 only)
Reduced to global top-10 most frequent next-layer experts per layer. 3113 H2D starts/token.  
**Result: Neutral (-2%, 1.323 tok/s).**  
**Root cause**: Even top-10 global prefetch didn't reduce actual misses enough to offset the H2D traffic. PCIe is the ceiling regardless.

### Trained MLP Predictor — 1,740 Samples
CrossLayerExpertPredictor: (h_normed[L] 4096-dim + layer_emb 32-dim) → 512-way multilabel.  
Trained on 1 prompt × 30 tokens = 1,740 (layer, hidden_state, next_experts) pairs.  
**Result**: 29.8% hit@10.

### Trained MLP Predictor — 34,800 Samples (More Data)
Collected 20 diverse prompts × 30 tokens each for better generalization.  
**Result: 25.7% hit@10 — worse than the smaller dataset.**  
**Root cause**: Cross-layer prediction (h[L] → experts[L+1]) is fundamentally hard. Attention + expert compute + residual + norm between layers makes h[L] a weak predictor of L+1 routing. More diverse prompts made the mapping harder, not easier. Not enough signal to generalize.

### Quantum Neural Network (considered, not implemented)
QNN proposed as alternative predictor.  
**Decision**: Skipped. QNN would face the same fundamental problem — h[L] simply doesn't contain enough information about L+1 routing regardless of model capacity. The input-output relationship is too many transformations removed.

---

## Bugs Discovered and Fixed

### Consume Order Bug
`avail_pf` (prefetch buffer) was checked before `gpu_cache`, meaning H2D copies were used even when the expert was already in VRAM.  
**Fix**: Consume order changed to `gpu_cache → avail_pf → hot_cache`.

### Produce Prefetch: GPU Cache Exclusion Bug
`produce_prefetch` was starting H2D transfers for experts already in `gpu_cache`.  
**Fix**: Added `key not in gpu_cache` guard before issuing H2D.

### Layer Boundary Device Mismatch
At layer 23→24 boundary, prefetch used layer L's device (cuda:0) for layer L+1 experts (which belong on cuda:1).  
**Fix**: `next_device = 'cuda:0' if next_layer < 24 else 'cuda:1'`

---

## Fundamental Limit

**PCIe bandwidth is destiny on discrete GPU + CPU systems.**

- PCIe Gen3 x16: ~13 GB/s CPU→GPU (H2D)
- Per token: 10 experts × 60 layers × 3.38MB = 2.028 GB minimum if all cold
- Even with 100% cache hit on HOT+CPU tiers, SSD misses still hit PCIe
- No amount of prediction accuracy overcomes 13 GB/s if you have to move data

Apple Silicon's 400 GB/s unified memory eliminates this wall entirely:
- Mac M3 Max (48GB, unified): **4.4 tok/s** on 397B (reference, not measured by this project)
- Dual RTX 3060 (24GB VRAM + 96GB DDR4, PCIe): **1.35 tok/s** on 397B

**Conclusion**: For large MoE on consumer hardware, unified memory architecture (Apple Silicon) dominates discrete GPU+PCIe regardless of software optimization depth.
