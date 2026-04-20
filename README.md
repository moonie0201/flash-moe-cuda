# 397B MoE Inference on Consumer Hardware

Running **Qwen3.5-397B-A17B** (397 billion parameters) on a dual RTX 3060 desktop at **1.35 tok/s** — 17% faster than llama.cpp on the same hardware.

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
| 3-tier cache, K=10 | 1.35 | +17% vs llama.cpp |
| **3-tier cache, K=4** | **1.80** | **+57% vs llama.cpp (quality trade-off)** |

## How It Works

The 397B model has 60 layers (45 linear attention + 15 full attention) × 512 experts per layer. Each token activates only 10 experts — but that's still 600 expert loads per token from a 120GB file on SSD.

### 3-Tier Expert Cache

```
Token → routing → which 10 experts needed?
         │
         ├─ GPU VRAM   (6.8GB,  ~1980 experts)  57% hit  ← instant
         ├─ CPU RAM    (30GB,   ~9216 experts)   43% hit  ← H2D 0.26ms
         └─ SSD        (120GB,  rest)             0.1%    ← 1.3ms
```

### Key Optimizations

- **2-bit quantized expert weights** — 120GB total (vs 480GB at BF16); 3.38MB per expert (repacked from 4-bit)
- **Float8 non-expert weights** — freed 7.64GB VRAM (19.6GB → 12.0GB)
- **Layer-split across GPUs** — layers 0–23 on GPU0, 24–59 on GPU1
- **Fused CUDA kernel** — 2-bit dequant + batched GEMV + SiLU in one launch
- **Thread pool I/O** — 16 parallel workers for SSD reads
- **Activation trace** — COLD pass builds expert frequency map for cache population

### The Hard Limit

PCIe Gen3 = 13 GB/s CPU→GPU. Every optimization runs into this wall eventually. Prediction, prefetch, compression — all tested, all hit the same ceiling. See [ses/EXPERIMENTS.md](ses/EXPERIMENTS.md) for the full log.

Apple Silicon (400 GB/s unified memory) eliminates this wall entirely — reference implementations on Mac M3 Max report 4.4 tok/s on the same model.

## Quick Start

```bash
cd ses

# 1. Repack model to 2-bit (one-time, ~1 hour)
python src/repack_397b_2bit.py --model /path/to/Qwen3.5-397B-A17B

# 2. Build CUDA extension
cd src && python setup_expert_ops.py build_ext --inplace && cd ..

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

## Structure

```
ses/
  src/
    run_397b_ssd.py          # Main inference script
    two_bit_loader.py        # 2-bit expert loader
    expert_ops.cu            # Fused CUDA kernel
    float8_nonexpert.py      # Float8 non-expert quantization
    repack_397b_2bit.py      # 4-bit → 2-bit repack
  README.md                  # SES-specific details
  EXPERIMENTS.md             # Full experiment log

paper/
  main.tex                   # Technical writeup
  references.bib

experiments/
  progress_log.md            # Benchmark results over time
  activation_profile_397B.json
```
