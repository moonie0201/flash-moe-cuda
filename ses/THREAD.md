# Twitter/X Thread

---

**[1/9]**
I ran a 397 billion parameter LLM on two $300 gaming GPUs.

Not a flex. It took 3 months, 90+ failed experiments, and I still only get 1.35 tok/s at full quality.

Here's what I learned. 🧵

---

**[2/9]**
The model: Qwen3.5-397B-A17B (Mixture of Experts)
- 60 layers, 512 experts per layer
- Only 10 experts activate per token
- 2-bit quantized: 120GB on SSD, 3.38MB per expert

Hardware: 2× RTX 3060 12GB (~$600 total)
Total VRAM: 24GB. The model needs 120GB.

How do you fit 120GB into 24GB?

---

**[3/9]**
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
SSD       (120GB)  →  0.1% miss    ← 1.3ms
```

Result: **1.35 tok/s** (+17% vs llama.cpp) ✓

---

**[4/9]**
To fit non-expert weights in 24GB across 2 GPUs:
- Split layers: GPU0 handles layers 0-23, GPU1 handles 24-59
- Float8 quantize attention weights → freed 7.64GB VRAM
- Custom CUDA kernel: fused 2-bit dequant + batched GEMV + SiLU in one pass

The kernel alone was worth writing. Replacing 10 serial expert calls with 1 batched CUDA launch.

---

**[5/9]**
Speed mode: K=4 instead of K=10

The Mac Metal implementation uses K=4 active experts for speed (vs model default K=10).

K=10 → 600 expert loads per token
K=4  → 240 expert loads per token (60% less)

Since PCIe is the bottleneck, fewer loads = faster token:

**1.80 tok/s at K=4 (+57% vs llama.cpp)**

Trade-off: quality is reduced but still coherent for general use.

---

**[6/9]**
Things I tried that FAILED:

❌ Cross-layer n-gram prefetch → -21% (flooded PCIe bandwidth)
❌ Trained MLP predictor (34K samples) → 25.7% hit@10 (worse with more data)
❌ FATE gate prediction → GPU hit rate 59.9%→76.6% ✓ but throughput -5% ✗
❌ torch.compile → +0.75% (GPU compute wasn't the bottleneck anyway)

The bottleneck is always PCIe (13 GB/s CPU→GPU).
Even accurate prediction adds H2D traffic that cancels the gain.
You can't outrun physics.

---

**[7/9]**
Throughput summary (2-bit quantized, dual RTX 3060):

Cold (no cache), K=10: 0.77 tok/s
GPU cache only, K=10: 1.09 tok/s
3-tier cache, K=10: 1.35 tok/s
3-tier cache, K=4: 1.80 tok/s (quality trade-off)

(Mac M3 Max unified memory: 4.4 tok/s. Unified memory eliminates the PCIe bottleneck entirely.)

---

**[8/9]**
The real lesson:

On discrete GPU + PCIe, **memory bandwidth is destiny**.

Every clever trick (prefetch, prediction, compression) runs into the same wall:
13 GB/s PCIe × expert size = hard ceiling

Apple Silicon wins because 400 GB/s unified memory eliminates the wall entirely.

For large MoE on discrete GPU: more VRAM or more RAM is more effective than better software.

---

**[9/9]**
Full code + 90 experiment log on GitHub:
→ github.com/moonie0201/flash-moe-cuda

Includes:
- 2-bit expert streaming engine
- Custom CUDA fused kernel
- Float8 attention quantization
- K=4 speed mode
- FATE gate prediction (and why it didn't help)
- Everything that failed (and why)

If you're trying to run large MoE models on consumer hardware, hopefully this saves you 3 months.

RT if you found this useful 🙏

#LLM #MachineLearning #CUDA #OpenSource
