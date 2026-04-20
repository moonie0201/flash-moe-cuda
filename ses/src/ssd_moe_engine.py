#!/usr/bin/env python3
"""SSD MoE Streaming Engine — expert를 SSD에서 직접 스트리밍.

transformers/accelerate를 우회. 직접 weight를 로드하고 forward pass 구현.

Non-expert (4.4GB): GPU에 상주
Expert (750GB bf16 / 220GB GPTQ): SSD에서 필요한 것만 로드

Usage:
    python ses/src/ssd_moe_engine.py --model-dir /home/mh/models/Qwen3.5-397B-A17B \
        --prompt "What is MoE?" --tokens 10
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


def load_tensor_mmap(mm, header, data_offset, key, device='cpu', dtype=torch.bfloat16):
    """Load a single tensor from mmap'd safetensors file."""
    meta = header[key]
    start, end = meta['data_offsets']
    raw = mm[data_offset + start:data_offset + end]
    shape = meta['shape']

    if meta['dtype'] == 'BF16':
        u16 = np.frombuffer(raw, dtype=np.uint16)
        t = torch.from_numpy(u16.copy()).view(torch.bfloat16).reshape(shape)
    elif meta['dtype'] == 'F32':
        arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        t = torch.from_numpy(arr.copy())
    elif meta['dtype'] == 'I32':
        arr = np.frombuffer(raw, dtype=np.int32).reshape(shape)
        t = torch.from_numpy(arr.copy())
    else:
        raise ValueError(f"Unknown dtype: {meta['dtype']}")

    return t.to(device=device, dtype=dtype) if dtype and meta['dtype'] != 'I32' else t.to(device)


class ShardManager:
    """Manages mmap'd safetensors shards."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
            self.index = json.load(f)
        self._shards = {}  # shard_name -> (mm, fd, data_offset, header)

    def _open(self, shard_name):
        if shard_name not in self._shards:
            path = os.path.join(self.model_dir, shard_name)
            fd = os.open(path, os.O_RDONLY)
            size = os.fstat(fd).st_size
            mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
            hs = struct.unpack('<Q', mm[:8])[0]
            hdr = json.loads(mm[8:8+hs].decode('utf-8'))
            self._shards[shard_name] = (mm, fd, 8+hs, hdr)
        return self._shards[shard_name]

    def load(self, key, device='cpu', dtype=torch.bfloat16):
        shard = self.index['weight_map'][key]
        mm, fd, do, hdr = self._open(shard)
        return load_tensor_mmap(mm, hdr, do, key, device, dtype)

    def load_expert_slice(self, key, expert_id, device='cpu'):
        """Load one expert from a [num_experts, ...] tensor."""
        shard = self.index['weight_map'][key]
        mm, fd, do, hdr = self._open(shard)
        meta = hdr[key]
        shape = meta['shape']  # [num_experts, d1, d2]
        start, end = meta['data_offsets']

        expert_elements = 1
        for d in shape[1:]:
            expert_elements *= d
        bytes_per = 2  # bf16
        offset = do + start + expert_id * expert_elements * bytes_per
        raw = mm[offset:offset + expert_elements * bytes_per]

        u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape[1:])
        return torch.from_numpy(u16.copy()).view(torch.bfloat16).to(device)

    def close(self):
        for mm, fd, _, _ in self._shards.values():
            mm.close()
            os.close(fd)


class RMSNorm:
    def __init__(self, weight, eps=1e-6):
        self.weight = weight
        self.eps = eps

    def __call__(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class SSDMoEEngine:
    """Full inference engine — non-expert on GPU, expert from SSD."""

    def __init__(self, model_dir, device='cuda:1'):
        self.shards = ShardManager(model_dir)
        self.device = torch.device(device)

        with open(os.path.join(model_dir, 'config.json')) as f:
            cfg = json.load(f)
        tc = cfg.get('text_config', cfg)
        self.num_layers = tc['num_hidden_layers']
        self.hidden_size = tc['hidden_size']
        self.num_experts = tc['num_experts']
        self.num_experts_per_tok = tc['num_experts_per_tok']
        self.moe_intermediate = tc['moe_intermediate_size']
        self.vocab_size = tc.get('vocab_size', 248320)

        print(f"Config: {self.num_layers}L, {self.num_experts}E, K={self.num_experts_per_tok}, "
              f"H={self.hidden_size}")

        # Load non-expert weights to GPU
        self._load_non_expert()

        # Expert stats
        self.expert_loads = 0

    def _get_layer_device(self, layer_idx):
        """Spread layers across available GPUs."""
        n_gpus = torch.cuda.device_count()
        if n_gpus >= 2:
            half = self.num_layers // 2
            return torch.device('cuda:0') if layer_idx < half else torch.device('cuda:1')
        return self.device

    def _load_non_expert(self):
        """Load all non-expert weights to GPUs (split across 2)."""
        print("Loading non-expert weights to GPUs...")
        t0 = time.time()
        S = self.shards

        # Embedding on cuda:0
        self.embed = S.load('model.language_model.embed_tokens.weight', 'cuda:0')
        print(f"  embed: {self.embed.shape} → cuda:0")

        # Final norm on cuda:1
        self.final_norm = RMSNorm(
            S.load('model.language_model.norm.weight', 'cuda:1')
        )

        # Per-layer weights — split across GPUs
        self.layers = []
        for i in range(self.num_layers):
            p = f'model.language_model.layers.{i}'
            D = self._get_layer_device(i)
            layer = {'device': D}

            # Input/post norms
            layer['input_norm'] = RMSNorm(S.load(f'{p}.input_layernorm.weight', D))
            layer['post_norm'] = RMSNorm(S.load(f'{p}.post_attention_layernorm.weight', D))

            # Attention (check type)
            wm = S.index['weight_map']
            if f'{p}.linear_attn.in_proj_qkv.weight' in wm:
                layer['attn_type'] = 'linear'
                layer['attn_qkv'] = S.load(f'{p}.linear_attn.in_proj_qkv.weight', D)
                layer['attn_z'] = S.load(f'{p}.linear_attn.in_proj_z.weight', D)
                layer['attn_out'] = S.load(f'{p}.linear_attn.out_proj.weight', D)
                layer['attn_a'] = S.load(f'{p}.linear_attn.in_proj_a.weight', D)
                layer['attn_b'] = S.load(f'{p}.linear_attn.in_proj_b.weight', D)
                layer['attn_conv'] = S.load(f'{p}.linear_attn.conv1d.weight', D)
                layer['attn_dt'] = S.load(f'{p}.linear_attn.dt_bias', D)
                layer['attn_A'] = S.load(f'{p}.linear_attn.A_log', D, dtype=torch.float32)
                layer['attn_norm'] = RMSNorm(
                    S.load(f'{p}.linear_attn.norm.weight', D, dtype=torch.float32))
            elif f'{p}.self_attn.q_proj.weight' in wm:
                layer['attn_type'] = 'full'
                layer['attn_q'] = S.load(f'{p}.self_attn.q_proj.weight', D)
                layer['attn_k'] = S.load(f'{p}.self_attn.k_proj.weight', D)
                layer['attn_v'] = S.load(f'{p}.self_attn.v_proj.weight', D)
                if f'{p}.self_attn.o_proj.weight' in wm:
                    layer['attn_o'] = S.load(f'{p}.self_attn.o_proj.weight', D)
                layer['q_norm'] = RMSNorm(S.load(f'{p}.self_attn.q_norm.weight', D))
                layer['k_norm'] = RMSNorm(S.load(f'{p}.self_attn.k_norm.weight', D))

            # Gate (routing)
            layer['gate'] = S.load(f'{p}.mlp.gate.weight', D)

            # Shared expert
            layer['shared_gate'] = S.load(f'{p}.mlp.shared_expert.gate_proj.weight', D)
            layer['shared_up'] = S.load(f'{p}.mlp.shared_expert.up_proj.weight', D)
            layer['shared_down'] = S.load(f'{p}.mlp.shared_expert.down_proj.weight', D)
            layer['shared_expert_gate'] = S.load(f'{p}.mlp.shared_expert_gate.weight', D)

            self.layers.append(layer)
            if i % 10 == 0:
                print(f"  layer {i}/{self.num_layers} loaded")
                sys.stdout.flush()

        elapsed = time.time() - t0
        gpu_mem = torch.cuda.memory_allocated(self.device) / 1e9
        print(f"  Done: {gpu_mem:.1f}GB on GPU, {elapsed:.0f}s")

    def _simplified_attention(self, hidden, layer, layer_idx):
        """Simplified attention — enough for inference demo.

        Full GatedDeltaNet/standard attention is complex.
        We use a simplified linear projection as approximation.
        """
        if layer['attn_type'] == 'linear':
            # Simplified linear attention
            # z output dim = 2*H, out_proj input dim = 2*H
            z = F.linear(hidden, layer['attn_z'])       # [B,S, 2*H]
            z_dim = z.shape[-1]
            z_gate, z_val = z[..., :z_dim//2], z[..., z_dim//2:]
            gated = torch.sigmoid(z_gate) * z_val       # [B,S, H]
            # out_proj: [H, 2*H] — need 2*H input, concat gated with itself
            out_input = torch.cat([gated, gated], dim=-1)  # [B,S, 2*H]
            out = F.linear(out_input, layer['attn_out'])   # [B,S, H]
            return out
        else:
            # Simplified full attention
            v = F.linear(hidden, layer['attn_v'])  # [B,S, v_dim]
            if 'attn_o' in layer:
                o_in_dim = layer['attn_o'].shape[1]  # o_proj input dim
                # Expand v to match o_proj input
                v_exp = v.repeat(1, 1, (o_in_dim // v.shape[-1]) + 1)[..., :o_in_dim]
                out = F.linear(torch.sigmoid(v_exp) * v_exp, layer['attn_o'])
            else:
                out = v[..., :self.hidden_size]
            return out

    def _moe_forward(self, hidden, layer, layer_idx):
        """MoE forward — experts loaded from SSD on demand."""
        B, S, H = hidden.shape
        h_flat = hidden.view(-1, H)

        # Gate routing
        scores = F.linear(h_flat, layer['gate'])
        probs = F.softmax(scores, dim=-1)
        topk_weights, topk_ids = torch.topk(probs, self.num_experts_per_tok, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Expert forward
        final = torch.zeros_like(h_flat)
        active_experts = topk_ids.unique().tolist()

        prefix = f'model.language_model.layers.{layer_idx}.mlp.experts'

        for eid in active_experts:
            mask = (topk_ids == eid).any(dim=-1)
            if not mask.any():
                continue

            token_ids = mask.nonzero(as_tuple=True)[0]
            expert_pos = (topk_ids[token_ids] == eid).nonzero(as_tuple=True)
            weights = topk_weights[token_ids, expert_pos[1]].unsqueeze(-1)
            current = h_flat[token_ids]

            # Load expert from SSD → same device as hidden
            expert_device = h_flat.device
            gu_w = self.shards.load_expert_slice(f'{prefix}.gate_up_proj', eid, expert_device)
            dw_w = self.shards.load_expert_slice(f'{prefix}.down_proj', eid, expert_device)
            self.expert_loads += 1

            # Expert computation
            gate_up = F.linear(current, gu_w)
            gate, up = gate_up.chunk(2, dim=-1)
            act = F.silu(gate) * up
            out = F.linear(act, dw_w)

            final[token_ids] += weights * out

            # Free GPU memory
            del gu_w, dw_w

        # Shared expert
        shared_gate = F.linear(h_flat, layer['shared_gate'])
        shared_up = F.linear(h_flat, layer['shared_up'])
        shared_act = F.silu(shared_gate) * shared_up
        shared_out = F.linear(shared_act, layer['shared_down'])
        shared_w = torch.sigmoid(F.linear(h_flat, layer['shared_expert_gate']))
        final = final + shared_w * shared_out

        return final.view(B, S, H)

    def forward_one_token(self, token_id, layer_timings=None):
        """Forward pass for one token through all layers."""
        hidden = self.embed[token_id].unsqueeze(0).unsqueeze(0)  # [1, 1, H]

        for i in range(self.num_layers):
            layer = self.layers[i]
            layer_dev = layer['device']
            t0 = time.perf_counter()

            # Move hidden to this layer's GPU if needed
            if hidden.device != layer_dev:
                hidden = hidden.to(layer_dev)

            # Attention
            residual = hidden
            hidden = layer['input_norm'](hidden)
            hidden = residual + self._simplified_attention(hidden, layer, i)

            # MoE
            residual = hidden
            hidden = layer['post_norm'](hidden)
            hidden = residual + self._moe_forward(hidden, layer, i)

            if layer_timings is not None:
                layer_timings.append(time.perf_counter() - t0)

        hidden = hidden.to('cuda:1')
        hidden = self.final_norm(hidden)
        return hidden

    def generate(self, token_ids, max_new_tokens=10):
        """Generate tokens autoregressively."""
        # Simple greedy: process last token only (no KV cache for simplicity)
        generated = list(token_ids)

        for t in range(max_new_tokens):
            t0 = time.perf_counter()
            timings = []

            with torch.no_grad():
                hidden = self.forward_one_token(generated[-1], timings)
                # LM head = embed^T (weight tying) — embed is on cuda:0
                logits = F.linear(hidden.squeeze().to(self.embed.device), self.embed)
                next_token = logits.argmax(dim=-1).item()

            elapsed = time.perf_counter() - t0
            generated.append(next_token)
            print(f"  token {t}: id={next_token}, {elapsed:.2f}s, "
                  f"expert_loads={self.expert_loads}")
            sys.stdout.flush()
            self.expert_loads = 0

        return generated[len(token_ids):]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--prompt', default='What is MoE?')
    parser.add_argument('--tokens', type=int, default=5)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    engine = SSDMoEEngine(args.model_dir, args.device)

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
    input_ids = tokenizer.encode(text)
    print(f"\nPrompt: {args.prompt} ({len(input_ids)} tokens)")

    print(f"Generating {args.tokens} tokens...\n")
    t0 = time.time()
    new_ids = engine.generate(input_ids, args.tokens)
    elapsed = time.time() - t0

    text_out = tokenizer.decode(new_ids, skip_special_tokens=True)
    print(f"\n=== {len(new_ids)} tokens in {elapsed:.1f}s ({len(new_ids)/elapsed:.2f} tok/s) ===")
    print(f"Output: {text_out}")

    engine.shards.close()


if __name__ == '__main__':
    main()
