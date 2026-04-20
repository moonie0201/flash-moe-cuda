#!/usr/bin/env python3
"""397B GPTQ-Int4 inference — improvement 2.

GPTQ-Int4 stores experts at ~6.5MB each (4× smaller than bf16 25MB).
With HOT 30% cache: ~55GB pinned RAM (fits in 125GB).

Target: 0.1-0.5 tok/s (vs bf16 baseline 0.030)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir',
                        default='/home/mh/models/Qwen3.5-397B-A17B-GPTQ-Int4')
    parser.add_argument('--prompt', default='What is MoE? One sentence.')
    parser.add_argument('--tokens', type=int, default=5)
    parser.add_argument('--split', type=int, default=20)
    args = parser.parse_args()

    print(f"=== 397B GPTQ-Int4 Benchmark ===", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    print("Loading 397B GPTQ to CPU...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    print(f"  Loaded in {time.time()-t0:.0f}s", flush=True)

    import psutil
    ram = psutil.virtual_memory()
    print(f"  RAM: {ram.used/1e9:.0f}GB used / {ram.total/1e9:.0f}GB", flush=True)

    # Inspect: how does GPTQ model store experts?
    print("\nChecking model structure...", flush=True)
    layer0 = model.model.layers[0]
    print(f"  layer0.mlp type: {type(layer0.mlp).__name__}")
    if hasattr(layer0.mlp, 'experts'):
        experts = layer0.mlp.experts
        print(f"  experts type: {type(experts).__name__}")
        # List attributes
        attrs = [a for a in dir(experts) if not a.startswith('_')]
        relevant = [a for a in attrs if 'proj' in a.lower() or 'weight' in a.lower() or 'expert' in a.lower()]
        print(f"  expert attrs: {relevant[:15]}")

        # Check if experts is a ModuleList (GPTQ) or Qwen3_5MoeExperts (packed)
        if hasattr(experts, '__iter__'):
            first_expert = list(experts)[0] if isinstance(experts, list) or hasattr(experts, '__getitem__') else None
            if first_expert is not None:
                try:
                    first = experts[0]
                    print(f"  experts[0] type: {type(first).__name__}")
                    sub_attrs = [a for a in dir(first) if not a.startswith('_') and 'proj' in a.lower()]
                    print(f"  experts[0] attrs: {sub_attrs}")
                    if hasattr(first, 'gate_proj'):
                        print(f"  experts[0].gate_proj type: {type(first.gate_proj).__name__}")
                except Exception as e:
                    print(f"  Can't iterate: {e}")

    # Split model across GPUs
    split = args.split
    def dev(name):
        if 'layers.' in name:
            return 'cuda:0' if int(name.split('layers.')[1].split('.')[0]) < split else 'cuda:1'
        if name.startswith('model.model.norm') or name.startswith('model.norm'):
            return 'cuda:1'
        return 'cuda:0'

    print("\nMoving non-expert to GPUs...", flush=True)
    moved_params = 0
    for n, p in model.named_parameters():
        if 'experts' not in n:
            try:
                p.data = p.data.to(dev(n))
                moved_params += 1
            except Exception as e:
                print(f"  Failed: {n}: {e}")
    for n, b in model.named_buffers():
        if 'experts' not in n and b is not None:
            try:
                b.data = b.data.to(dev(n))
            except:
                pass

    print(f"  Moved {moved_params} non-expert params", flush=True)
    for i in range(2):
        print(f"  GPU {i}: {torch.cuda.memory_allocated(i)/1e9:.1f}GB", flush=True)

    # Cross-GPU hooks
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

    print("\nGenerating...", flush=True)
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

    print(f"\n=== Results ===")
    print(f"  Model: 397B GPTQ-Int4")
    print(f"  Tokens: {nt} in {elapsed:.1f}s = {nt/elapsed:.3f} tok/s")
    print(f"  Output: {resp}")

    result = {
        'model': '397B GPTQ-Int4',
        'tokens': nt,
        'elapsed_s': elapsed,
        'tok_per_s': nt / elapsed,
        'output': resp,
    }
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/397b_gptq.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to experiments/397b_gptq.json")


if __name__ == '__main__':
    main()
