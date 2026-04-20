# 397B MoE on Dual RTX 3060 — Linux/CUDA

Running **Qwen3.5-397B-A17B** (397 billion parameters) on two consumer GPUs at **1.35 tok/s** — 17% faster than llama.cpp on the same hardware.

## Hardware

| | |
|---|---|
| GPUs | 2× NVIDIA RTX 3060 12GB |
| Total VRAM | 24 GB |
| RAM | 96 GB DDR4 |
| Storage | NVMe SSD (~3.5 GB/s) |
| OS | Linux, CUDA 12.1 |

## Results

| Configuration | tok/s | Notes |
|---|---|---|
| llama.cpp (Q4_K_M, 3 GPU layers) | 1.15 | Baseline |
| SSD only (cold) | 0.77 | K=10, no cache |
| GPU cache only | 1.09 | K=10, 6.8GB VRAM cache |
| **3-tier cache, K=10** | **1.35** | **+17% vs llama.cpp** |
| **3-tier cache, K=4** | **1.80** | **+57% vs llama.cpp (quality trade-off)** |

## How It Works

The 397B model has 60 layers (45 linear attention + 15 full attention) × 512 experts. Only 10 experts activate per token — but that's still 600 expert loads per token from a 120GB file on SSD.

### 3-Tier Expert Cache

```
Token → routing → which 10 experts needed?
         │
         ├─ GPU VRAM   (6.8GB,  ~1980 experts)  57% hit  ← instant
         ├─ CPU RAM    (30GB,   ~9216 experts)   43% hit  ← H2D 0.26ms
         └─ SSD        (120GB,  rest)             0.1% miss ← 1.3ms
```

First pass (COLD) builds an expert frequency map from the prompt. Subsequent passes serve hits from VRAM or pinned RAM; only true misses touch SSD.

### Key Optimizations

| Technique | Impact |
|---|---|
| Float8 non-expert weights | −7.64GB VRAM (19.6GB → 12.0GB) |
| Layer-split: GPU0=layers 0–23, GPU1=layers 24–59 | Both GPUs utilized |
| Custom CUDA kernel: fused 2-bit dequant + batched GEMV + SiLU | Replaces 10 serial calls with 1 batched launch |
| Thread pool I/O (16 workers) | SSD reads issued in parallel |
| CUDA events for H2D lifetime tracking | Zero stalls on CPU ref release |

### What I Tried — [full experiment log](EXPERIMENTS.md)

| Approach | Result |
|---|---|
| Float8 non-expert weights | ✓ −7.64GB VRAM freed |
| GPU hot cache (57% hit rate) | ✓ Core of the speedup |
| CPU pinned cache (43% hit rate) | ✓ Covers most remaining misses |
| Cross-layer n-gram prefetch | ✗ −21% (PCIe congestion) |
| N-gram reduced to top-10 global | ✗ Neutral (−2%) — PCIe still ceiling |
| Trained MLP predictor (34K samples) | ✗ 25.7% hit@10 — worse with more data |
| torch.compile | ✗ +0.75% — GPU wasn't the bottleneck |

**The hard limit**: PCIe Gen3 = 13 GB/s CPU→GPU. Every optimization eventually runs into this wall. Prediction, prefetch, compression — all hit the same ceiling.

## Quick Start

```bash
# 1. Repack model to 2-bit (one-time, ~1 hour)
python src/repack_397b_2bit.py --model /path/to/Qwen3.5-397B-A17B

# 2. Build CUDA extension
cd src && python setup_expert_ops.py build_ext --inplace

# 3. Run
python src/run_397b_ssd.py \
  --2bit --float8-nonexpert \
  --gpu-hot-gb 10 \
  --hot-pct 1.0 \
  --tokens 50 \
  --prompt "What is artificial intelligence?"
```

## Requirements

```
torch >= 2.5 (CUDA 12.1)
transformers >= 4.51
numpy, psutil
CUDA toolkit
```

## Files

```
src/
  run_397b_ssd.py          # Main inference script
  two_bit_loader.py        # 2-bit expert loader
  expert_ops.cu            # Fused CUDA kernel (dequant+GEMV+SiLU)
  float8_nonexpert.py      # Float8 non-expert weight quantization
  gpu_predictor.py         # Expert routing predictor (experimental)
  train_cross_predictor.py # MLP predictor training
  repack_397b_2bit.py      # 4-bit → 2-bit repack
```
