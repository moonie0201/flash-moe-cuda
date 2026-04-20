#!/usr/bin/env python3
"""Linux MoE SSD Streaming Engine for 397B inference.

Loads non-expert weights (4.4GB) to GPU, streams experts from SSD.
Monkey-patches transformers' MoE forward to use FAEC caching.

Usage:
    python ses/src/ssd_engine.py --model-dir /home/mh/models/Qwen3.5-397B-A17B \
        --prompt "What is MoE?" --tokens 30
"""
import argparse
import json
import mmap
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSDExpertProvider:
    """Provides expert weights from SSD via mmap + HOT cache.

    Replaces the in-memory expert tensors with on-demand SSD loading.
    """

    def __init__(self, model_dir, num_layers, num_experts, device, dtype=torch.bfloat16):
        self.model_dir = model_dir
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.device = device
        self.dtype = dtype

        # Load index
        with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
            self.index = json.load(f)

        # Cache for safetensors headers
        self._headers = {}
        self._mmaps = {}

        # HOT cache: (layer, expert_id) -> (gate_up_tensor, down_tensor)
        self.hot_cache = {}
        self.stats = {'hits': 0, 'misses': 0, 'loads': 0}

    def _ensure_mmap(self, shard_name):
        if shard_name not in self._mmaps:
            path = os.path.join(self.model_dir, shard_name)
            fd = os.open(path, os.O_RDONLY)
            file_size = os.fstat(fd).st_size
            mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)

            header_size = struct.unpack('<Q', mm[:8])[0]
            header = json.loads(mm[8:8 + header_size].decode('utf-8'))

            self._mmaps[shard_name] = (mm, fd, 8 + header_size, header)
        return self._mmaps[shard_name]

    def _load_tensor_from_shard(self, key):
        """Load a single tensor via mmap."""
        shard_name = self.index['weight_map'][key]
        mm, fd, data_offset, header = self._ensure_mmap(shard_name)
        meta = header[key]
        start, end = meta['data_offsets']

        raw = mm[data_offset + start:data_offset + end]

        if meta['dtype'] == 'BF16':
            u16 = np.frombuffer(raw, dtype=np.uint16)
            f32_bits = u16.astype(np.uint32) << 16
            arr = np.frombuffer(f32_bits.tobytes(), dtype=np.float32).reshape(meta['shape'])
            return torch.from_numpy(arr.copy()).to(self.dtype).to(self.device)
        elif meta['dtype'] == 'F32':
            arr = np.frombuffer(raw, dtype=np.float32).reshape(meta['shape'])
            return torch.from_numpy(arr.copy()).to(self.dtype).to(self.device)
        else:
            # For GPTQ int32 etc, load as-is
            arr = np.frombuffer(raw, dtype=np.int32).reshape(meta['shape'])
            return torch.from_numpy(arr.copy()).to(self.device)

    def load_expert(self, layer, expert_id):
        """Load one expert's gate_up and down weights.

        Returns (gate_up, down) tensors on device.
        """
        key = (layer, expert_id)
        if key in self.hot_cache:
            self.stats['hits'] += 1
            return self.hot_cache[key]

        self.stats['misses'] += 1
        self.stats['loads'] += 1

        # Load from safetensors via mmap
        gu_key = f'model.language_model.layers.{layer}.mlp.experts.gate_up_proj'
        dw_key = f'model.language_model.layers.{layer}.mlp.experts.down_proj'

        # These are [num_experts, dim1, dim2] tensors — load just one expert slice
        gu_shard = self.index['weight_map'][gu_key]
        dw_shard = self.index['weight_map'][dw_key]

        mm_gu, _, do_gu, hdr_gu = self._ensure_mmap(gu_shard)
        mm_dw, _, do_dw, hdr_dw = self._ensure_mmap(dw_shard)

        meta_gu = hdr_gu[gu_key]
        meta_dw = hdr_dw[dw_key]

        # Calculate per-expert offset
        gu_shape = meta_gu['shape']  # [512, 2048, 4096] or similar
        dw_shape = meta_dw['shape']

        expert_gu_elements = gu_shape[1] * gu_shape[2]
        expert_dw_elements = dw_shape[1] * dw_shape[2]
        bytes_per = 2  # bf16

        gu_start, _ = meta_gu['data_offsets']
        dw_start, _ = meta_dw['data_offsets']

        gu_offset = do_gu + gu_start + expert_id * expert_gu_elements * bytes_per
        dw_offset = do_dw + dw_start + expert_id * expert_dw_elements * bytes_per

        gu_raw = mm_gu[gu_offset:gu_offset + expert_gu_elements * bytes_per]
        dw_raw = mm_dw[dw_offset:dw_offset + expert_dw_elements * bytes_per]

        # bf16 → tensor
        gu_u16 = np.frombuffer(gu_raw, dtype=np.uint16).reshape(gu_shape[1], gu_shape[2])
        dw_u16 = np.frombuffer(dw_raw, dtype=np.uint16).reshape(dw_shape[1], dw_shape[2])

        gate_up = torch.from_numpy(gu_u16.copy()).view(torch.bfloat16).to(self.device)
        down = torch.from_numpy(dw_u16.copy()).view(torch.bfloat16).to(self.device)

        return gate_up, down

    def warmup_hot_cache(self, hot_ids_per_layer):
        """Pre-load HOT experts into RAM (pinned memory for fast GPU transfer)."""
        total = 0
        t0 = time.time()
        for layer, expert_ids in hot_ids_per_layer.items():
            for eid in expert_ids:
                gu, dw = self.load_expert(layer, eid)
                # Move to CPU pinned for fast transfer
                self.hot_cache[(layer, eid)] = (
                    gu.cpu().pin_memory(),
                    dw.cpu().pin_memory(),
                )
                total += 1
        self.stats = {'hits': 0, 'misses': 0, 'loads': 0}  # Reset after warmup
        elapsed = time.time() - t0
        print(f"  HOT cache: {total} experts in {elapsed:.0f}s")

    def print_stats(self):
        total = self.stats['hits'] + self.stats['misses']
        if total > 0:
            print(f"  Cache: {self.stats['hits']}/{total} hits "
                  f"({self.stats['hits']/total:.1%}), "
                  f"{self.stats['loads']} SSD loads")


def patch_moe_forward(model, expert_provider, act_fn):
    """Monkey-patch all MoE layers to use SSD streaming."""

    def make_ssd_expert_forward(layer_idx, provider):
        def ssd_expert_forward(self, hidden_states, top_k_index, top_k_weights):
            final_hidden_states = torch.zeros_like(hidden_states)
            device = hidden_states.device

            with torch.no_grad():
                expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_idx in expert_hit:
                expert_idx = expert_idx[0].item()
                if expert_idx == self.num_experts:
                    continue

                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]

                # Load expert from SSD/cache → move to same device as hidden_states
                gate_up_w, down_w = provider.load_expert(layer_idx, expert_idx)
                gate_up_w = gate_up_w.to(device, non_blocking=True)
                down_w = down_w.to(device, non_blocking=True)

                # Expert forward
                gate, up = F.linear(current_state, gate_up_w).chunk(2, dim=-1)
                current_hidden_states = act_fn(gate) * up
                current_hidden_states = F.linear(current_hidden_states, down_w)
                current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(
                    0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

            return final_hidden_states
        return ssd_expert_forward

    # Find and patch each MoE layer
    for name, module in model.named_modules():
        if type(module).__name__ == 'Qwen3_5MoeExperts':
            # Extract layer index from name
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    break

            # Remove accelerate hooks that try to move expert weights to GPU
            if hasattr(module, '_hf_hook'):
                delattr(module, '_hf_hook')
            if hasattr(module, '_old_forward'):
                # accelerate wraps forward in hooks; we need to bypass
                pass

            # Bind the new forward directly (bypassing accelerate hooks)
            bound_fn = make_ssd_expert_forward(layer_idx, expert_provider)
            # Replace the forward at class instance level
            import types
            module.forward = types.MethodType(bound_fn, module)

            # Remove original weight tensors to free memory
            if hasattr(module, 'gate_up_proj') and module.gate_up_proj is not None:
                del module.gate_up_proj
                module.register_buffer('gate_up_proj', None)
            if hasattr(module, 'down_proj') and module.down_proj is not None:
                del module.down_proj
                module.register_buffer('down_proj', None)
            print(f"  Patched layer {layer_idx} experts → SSD streaming")


def main():
    parser = argparse.ArgumentParser(description='397B MoE SSD Streaming Inference')
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--prompt', default='What is a Mixture of Experts model? Explain briefly.')
    parser.add_argument('--tokens', type=int, default=20)
    parser.add_argument('--hot-pct', type=float, default=0.0,
                        help='HOT cache percentage (0=no cache, 0.5=50%%)')
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    tc = config.text_config
    print(f"Model: {tc.num_hidden_layers} layers, {tc.num_experts} experts, "
          f"K={tc.num_experts_per_tok}, hidden={tc.hidden_size}")

    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    print("\n2. Loading model skeleton (meta tensors, no expert weights)...")
    # Custom device map: everything to GPU except experts
    device_map = {}
    n_layers = tc.num_hidden_layers

    # Strategy: let accelerate auto-distribute with max_memory constraints.
    # Key: experts go to CPU, non-expert to GPU.
    # We use max_memory to limit GPU usage → forces large experts to CPU.
    # Then monkey-patch expert forward to use our SSD streaming.
    device_map = "auto"
    max_memory = {
        0: "10GiB",  # GPU 0: some attention layers
        1: "10GiB",  # GPU 1: some attention layers
        "cpu": "110GiB",  # CPU: experts land here
    }

    t0 = time.time()
    offload_dir = '/tmp/ssd_engine_offload'
    os.makedirs(offload_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=offload_dir,
        trust_remote_code=True,
    )
    print(f"  Loaded in {time.time()-t0:.0f}s")

    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.memory_allocated(i)/1e9:.1f}GB")

    print("\n3. Setting up SSD expert provider...")
    provider = SSDExpertProvider(
        args.model_dir, n_layers, tc.num_experts,
        device='cuda:1', dtype=torch.bfloat16
    )

    # Get activation function from model
    act_fn = nn.SiLU()

    print("\n4. Patching MoE layers for SSD streaming...")
    patch_moe_forward(model, provider, act_fn)

    # Optional: warmup HOT cache
    if args.hot_pct > 0:
        print(f"\n5. Warming HOT cache ({args.hot_pct:.0%})...")
        # Quick profiling
        gate_shard = 'model.safetensors-00094-of-00094.safetensors'
        from safetensors.torch import load_file
        gate_tensors = load_file(os.path.join(args.model_dir, gate_shard))
        hot_ids = {}
        for layer in range(n_layers):
            key = f'model.language_model.layers.{layer}.mlp.gate.weight'
            if key in gate_tensors:
                W = gate_tensors[key].float().numpy()
                freq = np.zeros(tc.num_experts)
                for _ in range(500):
                    x = np.random.randn(tc.hidden_size).astype(np.float32)
                    s = W @ x; s -= s.max()
                    p = np.exp(s) / np.exp(s).sum()
                    freq[np.argsort(p)[-tc.num_experts_per_tok:]] += 1
                n_hot = int(tc.num_experts * args.hot_pct)
                hot_ids[layer] = np.argsort(freq)[-n_hot:].tolist()
        del gate_tensors
        provider.warmup_hot_cache(hot_ids)

    print(f"\n{'='*50}")
    print(f"Generating: {args.prompt}")
    print(f"{'='*50}")

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.tokens, do_sample=False)
    elapsed = time.time() - t0

    new_toks = out.shape[1] - inputs['input_ids'].shape[1]
    response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    print(f"\n=== {new_toks} tokens in {elapsed:.1f}s ({new_toks/elapsed:.2f} tok/s) ===")
    print(response)
    print()
    provider.print_stats()


if __name__ == '__main__':
    main()
