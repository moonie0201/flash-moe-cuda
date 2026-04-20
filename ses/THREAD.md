# Twitter/X Thread

---

**[1/8]**
I ran a 397 billion parameter LLM on two $300 gaming GPUs.

Not a flex. It took 3 months, 60+ failed experiments, and I still only get 1.35 tok/s.

Here's what I learned. 🧵

---

**[2/8]**
The model: Qwen3.5-397B-A17B (Mixture of Experts)
- 60 layers, 512 experts per layer
- Only 10 experts activate per token
- 2-bit quantized: 120GB on SSD, 3.38MB per expert

Hardware: 2× RTX 3060 12GB (~$600 total)
Total VRAM: 24GB. The model needs 120GB.

How do you fit 120GB into 24GB?

---

**[3/8]**
You don't. You stream it.

Every token → routing decides which 10 experts to use
→ load those 10 experts from SSD
→ compute
→ repeat for all 60 layers

That's 600 expert loads per token. Cold: **0.77 tok/s**

So I built a 3-tier cache:

```
GPU VRAM  (6.8GB)  → 57% hit rate  ← instant
CPU RAM   (30GB)   → 43% hit rate  ← 0.26ms H2D  
SSD       (120GB)  →  0.1% hit rate ← 1.3ms
```

Result: **1.35 tok/s** ✓

---

**[4/8]**
To fit non-expert weights in 24GB across 2 GPUs:
- Split layers: GPU0 handles layers 0-23, GPU1 handles 24-59
- Float8 quantize attention weights → freed 7.64GB VRAM
- Custom CUDA kernel: fused 2-bit dequant + batched GEMV + SiLU in one pass

The kernel alone was worth writing. Replacing 10 serial expert calls with 1 batched CUDA launch.

---

**[5/8]**
Things I tried that FAILED:

❌ Cross-layer n-gram prefetch → -21% (flooded PCIe bandwidth)
❌ Trained MLP predictor (34K samples) → 25.7% hit rate (not enough)  
❌ torch.compile → +0.75% (GPU compute wasn't the bottleneck anyway)
❌ Vectorized H2D → pinned memory + fancy indexing = extra copy overhead

The bottleneck is always PCIe (13 GB/s CPU→GPU).
You can't outrun physics.

---

**[6/8]**
How does it compare to llama.cpp?

llama.cpp with Q4_K_M + 3 GPU layers: **1.15 tok/s**
Our Python implementation: **1.35 tok/s** (+17%)

A Python script beating llama.cpp's C++ on the same hardware felt good.

(Mac M3 Max with unified memory gets 4.4 tok/s. Unified memory is cheating.)

---

**[7/8]**
The real lesson:

On discrete GPU + PCIe, **memory bandwidth is destiny**.

Every clever trick (prefetch, prediction, compression) runs into the same wall:
13 GB/s PCIe × expert size = hard ceiling

Apple Silicon wins because 400 GB/s unified memory eliminates the wall entirely.

For MoE on consumer hardware: **buy a Mac** or **upgrade VRAM**.

---

**[8/8]**
Full code + 60 experiment log on GitHub:
→ github.com/moonie0201/flash-moe-cuda

Includes:
- 2-bit expert streaming engine
- Custom CUDA fused kernel
- Float8 attention quantization
- Everything that failed (and why)

If you're trying to run large MoE models on consumer hardware, hopefully this saves you 3 months.

RT if you found this useful 🙏

#LLM #MachineLearning #CUDA #OpenSource
