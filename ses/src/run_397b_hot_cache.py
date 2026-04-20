#!/usr/bin/env python3
"""397B inference with HOT expert cache — improvement 1.

Baseline (no cache): 0.03 tok/s
Target with HOT 50%: ~0.1 tok/s (75% hit rate reduces CPU→GPU transfers)
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_hot_cache(model, hot_pct, n_experts, num_experts_per_tok, hidden_size, device_map):
    """Profile expert activation → pin hot experts in CPU pinned memory.

    Using CPU pinned memory (not GPU) since GPU VRAM is full with non-expert weights.
    Pinned memory gives 2-3× faster CPU→GPU transfer than pageable.

    Returns: {(layer, expert_id): (gate_up_pinned, down_pinned)}
    """
    print(f"Profiling HOT experts (top {hot_pct:.0%}) → CPU pinned memory...", flush=True)
    np.random.seed(42)
    n_samples = 500
    n_hot_per_layer = int(n_experts * hot_pct)

    hot_cache = {}
    total_bytes = 0

    for i, layer in enumerate(model.model.layers):
        gate_w = layer.mlp.gate.weight
        target_gpu = gate_w.device

        # Profile routing frequency
        freq = np.zeros(n_experts, dtype=np.float64)
        with torch.no_grad():
            for _ in range(n_samples):
                x = torch.randn(hidden_size, device=target_gpu, dtype=torch.bfloat16)
                scores = F.linear(x, gate_w)
                probs = torch.softmax(scores.float(), dim=0)
                topk_ids = torch.topk(probs, num_experts_per_tok).indices.cpu().numpy()
                freq[topk_ids] += 1
        hot_ids = np.argsort(freq)[-n_hot_per_layer:].tolist()

        experts_mod = layer.mlp.experts
        gu_all = experts_mod.gate_up_proj  # already CPU
        dw_all = experts_mod.down_proj

        # Pin HOT expert slices to pinned memory for fast GPU transfer
        for eid in hot_ids:
            gu = gu_all[eid].clone().pin_memory()
            dw = dw_all[eid].clone().pin_memory()
            hot_cache[(i, eid)] = (gu, dw)
            total_bytes += gu.nelement() * 2 + dw.nelement() * 2

        if i % 10 == 0 or i == len(model.model.layers) - 1:
            print(f"  layer {i+1}/{len(model.model.layers)} profiled, "
                  f"pinned: {total_bytes/1e9:.1f}GB", flush=True)

    print(f"HOT cache: {len(hot_cache)} experts, {total_bytes/1e9:.1f}GB (CPU pinned)",
          flush=True)
    return hot_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='/home/mh/models/Qwen3.5-397B-A17B')
    parser.add_argument('--hot-pct', type=float, default=0.3,
                        help='Fraction of experts to cache on GPU (default 0.3)')
    parser.add_argument('--prompt', default='What is MoE? One sentence.')
    parser.add_argument('--tokens', type=int, default=5)
    parser.add_argument('--split', type=int, default=20,
                        help='Layer split between GPU 0 and GPU 1')
    args = parser.parse_args()

    print(f"=== 397B HOT Cache Benchmark (hot_pct={args.hot_pct:.0%}) ===", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    print("Loading 397B to CPU...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    print(f"  Loaded in {time.time()-t0:.0f}s", flush=True)

    # Non-expert to GPUs
    split = args.split
    def dev(name):
        if 'layers.' in name:
            return 'cuda:0' if int(name.split('layers.')[1].split('.')[0]) < split else 'cuda:1'
        if name.startswith('model.model.norm') or name.startswith('model.norm'):
            return 'cuda:1'
        return 'cuda:0'

    print("Moving non-expert to GPUs...", flush=True)
    for n, p in model.named_parameters():
        if 'experts' not in n:
            p.data = p.data.to(dev(n))
    for n, b in model.named_buffers():
        if 'experts' not in n and b is not None:
            b.data = b.data.to(dev(n))

    for i in range(2):
        print(f"  GPU {i}: {torch.cuda.memory_allocated(i)/1e9:.1f}GB", flush=True)

    # Build HOT cache
    cfg = model.config.get_text_config() if hasattr(model.config, 'get_text_config') else model.config
    if not hasattr(cfg, 'num_experts'):
        cfg = model.config
    hot_cache = build_hot_cache(
        model, args.hot_pct,
        cfg.num_experts, cfg.num_experts_per_tok, cfg.hidden_size,
        None,
    )
    for i in range(2):
        print(f"  GPU {i} after HOT: {torch.cuda.memory_allocated(i)/1e9:.1f}GB", flush=True)

    # Stats tracking
    stats = {'hits': 0, 'misses': 0}

    # Hook non-expert cross-GPU transfer
    def mkhook(d):
        def h(m, args, kwargs):
            a = list(args)
            if a and isinstance(a[0], torch.Tensor):
                a[0] = a[0].to(d)
            for k in ['hidden_states', 'attention_mask']:
                if k in kwargs and kwargs[k] is not None and isinstance(kwargs[k], torch.Tensor):
                    kwargs[k] = kwargs[k].to(d)
            if 'position_embeddings' in kwargs and kwargs['position_embeddings'] is not None:
                kwargs['position_embeddings'] = tuple(
                    t.to(d) if isinstance(t, torch.Tensor) else t
                    for t in kwargs['position_embeddings']
                )
            return tuple(a), kwargs
        return h

    for i, layer in enumerate(model.model.layers):
        layer.register_forward_pre_hook(
            mkhook(torch.device('cuda:0' if i < split else 'cuda:1')),
            with_kwargs=True
        )

    def to_dev(d):
        def hook(m, args):
            a = list(args)
            if a and isinstance(a[0], torch.Tensor):
                a[0] = a[0].to(d)
            return tuple(a)
        return hook

    model.model.norm.register_forward_pre_hook(to_dev('cuda:1'))
    model.lm_head.register_forward_pre_hook(to_dev('cuda:0'))

    # Patched expert forward with HOT cache
    def make_expert_fwd(layer_idx):
        def fwd(self, hs, tki, tkw):
            d = hs.device
            tki = tki.to(d)
            tkw = tkw.to(d)
            out = torch.zeros_like(hs)
            with torch.no_grad():
                mask = F.one_hot(tki, num_classes=self.num_experts).permute(2, 1, 0)
                hits = mask.sum(dim=(-1, -2)).gt(0).nonzero()
            for ei in hits:
                e = ei[0].item()
                if e == self.num_experts:
                    continue
                p, t = torch.where(mask[e])

                # Check HOT cache (CPU pinned) or fallback to CPU pageable
                key = (layer_idx, e)
                if key in hot_cache:
                    gu_src, dw_src = hot_cache[key]
                    stats['hits'] += 1
                else:
                    gu_src = self.gate_up_proj[e]
                    dw_src = self.down_proj[e]
                    stats['misses'] += 1

                # pinned → GPU is faster with non_blocking=True
                gu = gu_src.to(d, non_blocking=True)
                dw = dw_src.to(d, non_blocking=True)

                g, u = F.linear(hs[t], gu).chunk(2, dim=-1)
                c = F.silu(g) * u
                c = F.linear(c, dw) * tkw[t, p, None]
                out.index_add_(0, t, c.to(out.dtype))
                del gu, dw
            return out
        return fwd

    for i, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        experts.forward = types.MethodType(make_expert_fwd(i), experts)

    print("Generating...", flush=True)
    msgs = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to('cuda:0')

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.tokens, do_sample=False)
    elapsed = time.time() - t0

    nt = out.shape[1] - inputs['input_ids'].shape[1]
    resp = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    total = stats['hits'] + stats['misses']
    hit_rate = stats['hits'] / total if total > 0 else 0.0

    print(f"\n=== Results ===")
    print(f"  Model: 397B bf16")
    print(f"  HOT cache: {args.hot_pct:.0%} = {len(hot_cache)} experts")
    print(f"  Cache hit rate: {hit_rate:.1%} ({stats['hits']}/{total})")
    print(f"  Tokens: {nt} in {elapsed:.1f}s = {nt/elapsed:.3f} tok/s")
    print(f"  Output: {resp}")

    # Save results
    result = {
        'model': '397B bf16',
        'hot_pct': args.hot_pct,
        'hot_experts': len(hot_cache),
        'hit_rate': hit_rate,
        'hits': stats['hits'],
        'misses': stats['misses'],
        'tokens': nt,
        'elapsed_s': elapsed,
        'tok_per_s': nt / elapsed,
        'output': resp,
    }
    out_path = f"experiments/397b_hot_{int(args.hot_pct*100)}.json"
    os.makedirs('experiments', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
