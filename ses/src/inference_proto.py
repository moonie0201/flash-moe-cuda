#!/usr/bin/env python3
"""SES Inference Prototype — PyTorch-based MoE inference with HOT cache + GPU prediction.

Runs Qwen3.5-35B-A3B on consumer hardware:
  GPU 0: Expert routing prediction (dedicated)
  GPU 1: Matrix computations (attention projections, expert forward)
  CPU:   Linear attention (BLAS), coordination
  RAM:   HOT expert cache (~15-50GB)

Usage:
    python ses/src/inference_proto.py --model-dir models/Qwen3.5-35B-A3B --mode fast
    python ses/src/inference_proto.py --model-dir models/Qwen3.5-35B-A3B --mode lossless
"""
import argparse
import json
import os
import time
import struct

import numpy as np
import torch
from safetensors.torch import load_file


class ModelConfig:
    """Parsed model configuration."""
    def __init__(self, config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        tc = cfg.get('text_config', cfg)
        self.num_layers = tc['num_hidden_layers']
        self.hidden_size = tc['hidden_size']
        self.moe_intermediate_size = tc['moe_intermediate_size']
        self.num_experts = tc['num_experts']
        self.num_experts_per_tok = tc['num_experts_per_tok']
        self.num_attention_heads = tc.get('num_attention_heads', 16)
        self.num_kv_heads = tc.get('num_key_value_heads', 2)
        self.vocab_size = tc.get('vocab_size', 248320)


class ExpertLoader:
    """Load expert weights from safetensors files."""

    def __init__(self, model_dir, config, mode='fast'):
        self.model_dir = model_dir
        self.config = config
        self.mode = mode  # 'fast' (bf16→compute) or 'lossless' (bf16 native)

        with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
            self.index = json.load(f)

        self._file_cache = {}  # shard_name -> loaded tensors

    def _load_shard(self, shard_name):
        if shard_name not in self._file_cache:
            path = os.path.join(self.model_dir, shard_name)
            self._file_cache[shard_name] = load_file(path)
        return self._file_cache[shard_name]

    def load_expert_gate_up(self, layer, expert_id):
        """Load gate_up_proj for one expert. Returns [1024, hidden_dim] float32."""
        key = f'model.language_model.layers.{layer}.mlp.experts.gate_up_proj'
        shard = self.index['weight_map'][key]
        tensors = self._load_shard(shard)
        W = tensors[key][expert_id].float()  # [1024, hidden_dim]
        return W

    def load_expert_down(self, layer, expert_id):
        """Load down_proj for one expert. Returns [hidden_dim, intermediate] float32."""
        key = f'model.language_model.layers.{layer}.mlp.experts.down_proj'
        shard = self.index['weight_map'][key]
        tensors = self._load_shard(shard)
        W = tensors[key][expert_id].float()  # [hidden_dim, intermediate]
        return W

    def unload_shard(self, shard_name):
        """Free a cached shard to reclaim memory."""
        if shard_name in self._file_cache:
            del self._file_cache[shard_name]


class HotExpertCache:
    """Frequency-aware expert cache — HOT experts pinned in RAM/GPU."""

    def __init__(self, config, device=None):
        self.config = config
        self.device = device or torch.device('cpu')
        self.cache = {}  # (layer, expert_id) -> {'gate_up': tensor, 'down': tensor}
        self.stats = {'hits': 0, 'misses': 0}

    def warmup(self, loader, hot_ids_per_layer, device=None):
        """Pre-load HOT experts.

        Args:
            loader: ExpertLoader instance
            hot_ids_per_layer: {layer: [expert_ids]}
        """
        dev = device or self.device
        total_bytes = 0
        t0 = time.time()
        for layer, expert_ids in hot_ids_per_layer.items():
            for eid in expert_ids:
                gate_up = loader.load_expert_gate_up(layer, eid).to(dev)
                down = loader.load_expert_down(layer, eid).to(dev)
                self.cache[(layer, eid)] = {'gate_up': gate_up, 'down': down}
                total_bytes += gate_up.nelement() * 4 + down.nelement() * 4
        elapsed = time.time() - t0
        n_experts = len(self.cache)
        print(f"HOT cache: {n_experts} experts, {total_bytes/1e9:.1f} GB, "
              f"loaded in {elapsed:.1f}s")

    def get(self, layer, expert_id):
        """Get expert weights. Returns (weights_dict, is_cached)."""
        key = (layer, expert_id)
        if key in self.cache:
            self.stats['hits'] += 1
            return self.cache[key], True
        self.stats['misses'] += 1
        return None, False

    def hit_rate(self):
        total = self.stats['hits'] + self.stats['misses']
        return self.stats['hits'] / total if total > 0 else 0.0


def swiglu(gate_up_out):
    """SwiGLU activation on concatenated gate+up output."""
    mid = gate_up_out.shape[0] // 2
    gate = gate_up_out[:mid]
    up = gate_up_out[mid:]
    return torch.nn.functional.silu(gate) * up


def expert_forward(gate_up_w, down_w, x):
    """Single expert forward: gate_up projection → SwiGLU → down projection."""
    gate_up_out = gate_up_w @ x
    act = swiglu(gate_up_out)
    return down_w @ act


def moe_forward(expert_weights_list, routing_weights, x, hidden_dim):
    """MoE forward: weighted sum of K expert outputs."""
    output = torch.zeros(hidden_dim, dtype=x.dtype, device=x.device)
    for k, (gate_up_w, down_w) in enumerate(expert_weights_list):
        expert_out = expert_forward(gate_up_w, down_w, x)
        output += routing_weights[k] * expert_out
    return output


def benchmark_moe_layer(config, loader, cache, gate_weight_gpu,
                        device_compute, layer_idx, n_tokens=50):
    """Benchmark one MoE layer with HOT cache + prediction."""
    hidden_dim = config.hidden_size
    num_experts = config.num_experts
    K = config.num_experts_per_tok

    timings = {
        'predict_us': [], 'routing_us': [], 'cache_check_us': [],
        'load_us': [], 'compute_us': [], 'total_us': [],
    }

    for t in range(n_tokens):
        x = torch.randn(hidden_dim, device=device_compute)
        t_total_start = time.perf_counter()

        # 1. GPU prediction (would be on GPU 0 in production, here same device)
        t0 = time.perf_counter()
        with torch.no_grad():
            scores = gate_weight_gpu @ x.to(gate_weight_gpu.device)
            probs = torch.softmax(scores, dim=0)
            topk_vals, topk_ids = torch.topk(probs, K)
            routing_weights = (topk_vals / topk_vals.sum()).to(device_compute)
        torch.cuda.synchronize()
        timings['predict_us'].append((time.perf_counter() - t0) * 1e6)

        # 2. Check cache + load experts
        expert_data = []
        t0 = time.perf_counter()
        for k in range(K):
            eid = topk_ids[k].item()
            cached, is_hit = cache.get(layer_idx, eid)
            if is_hit:
                expert_data.append((cached['gate_up'], cached['down']))
            else:
                # SSD load (simulated — load from safetensors)
                t_load = time.perf_counter()
                gu = loader.load_expert_gate_up(layer_idx, eid).to(device_compute)
                dw = loader.load_expert_down(layer_idx, eid).to(device_compute)
                expert_data.append((gu, dw))
                timings['load_us'].append((time.perf_counter() - t_load) * 1e6)
        timings['cache_check_us'].append((time.perf_counter() - t0) * 1e6)

        # 3. Expert forward
        t0 = time.perf_counter()
        with torch.no_grad():
            output = moe_forward(expert_data, routing_weights, x, hidden_dim)
        torch.cuda.synchronize()
        timings['compute_us'].append((time.perf_counter() - t0) * 1e6)

        timings['total_us'].append((time.perf_counter() - t_total_start) * 1e6)

    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['fast', 'lossless'], default='fast')
    parser.add_argument('--layer', type=int, default=0, help='Layer to benchmark')
    parser.add_argument('--tokens', type=int, default=50)
    parser.add_argument('--hot-pct', type=float, default=0.2, help='Fraction of experts to cache')
    args = parser.parse_args()

    config = ModelConfig(os.path.join(args.model_dir, 'config.json'))
    print(f"Model: {config.num_layers} layers, {config.num_experts} experts, "
          f"K={config.num_experts_per_tok}, hidden={config.hidden_size}")

    device_predict = torch.device('cuda:0')
    device_compute = torch.device('cuda:1') if torch.cuda.device_count() > 1 else torch.device('cuda:0')
    print(f"Prediction: {device_predict}, Compute: {device_compute}")

    # Load gate weight to GPU 0
    loader = ExpertLoader(args.model_dir, config, args.mode)
    # Look up gate weight shard from index (not hardcoded)
    index_path = os.path.join(args.model_dir, 'model.safetensors.index.json')
    with open(index_path) as f:
        weight_index = json.load(f)
    gate_key = f'model.language_model.layers.{args.layer}.mlp.gate.weight'
    gate_shard = weight_index['weight_map'][gate_key]
    gate_tensors = load_file(os.path.join(args.model_dir, gate_shard))
    gate_weight_gpu = gate_tensors[gate_key].float().to(device_predict)
    print(f"Gate weight on {device_predict}: {gate_weight_gpu.shape}")

    # Profile HOT experts
    print("Profiling expert activation frequency...")
    freq = np.zeros(config.num_experts)
    W_np = gate_weight_gpu.cpu().numpy()
    for _ in range(2000):
        x = np.random.randn(config.hidden_size).astype(np.float32)
        scores = W_np @ x
        scores -= scores.max()
        probs = np.exp(scores) / np.exp(scores).sum()
        freq[np.argsort(probs)[-config.num_experts_per_tok:]] += 1

    n_hot = int(config.num_experts * args.hot_pct)
    hot_ids = np.argsort(freq)[-n_hot:].tolist()
    print(f"HOT experts: {n_hot} (top {args.hot_pct:.0%})")

    # Warmup cache
    cache = HotExpertCache(config, device=device_compute)
    cache.warmup(loader, {args.layer: hot_ids}, device=device_compute)

    # Benchmark
    print(f"\nBenchmarking layer {args.layer}, {args.tokens} tokens...")
    timings = benchmark_moe_layer(
        config, loader, cache, gate_weight_gpu,
        device_compute, args.layer, args.tokens
    )

    # Results
    print(f"\n=== Results (Layer {args.layer}, {args.tokens} tokens) ===")
    print(f"  Prediction:  {np.mean(timings['predict_us']):>8.1f} µs (P50: {np.percentile(timings['predict_us'], 50):.1f})")
    print(f"  Cache+Load:  {np.mean(timings['cache_check_us']):>8.1f} µs")
    if timings['load_us']:
        print(f"  SSD Load:    {np.mean(timings['load_us']):>8.1f} µs ({len(timings['load_us'])} loads)")
    print(f"  Compute:     {np.mean(timings['compute_us']):>8.1f} µs")
    print(f"  Total/token: {np.mean(timings['total_us']):>8.1f} µs")
    print(f"  Cache hit:   {cache.hit_rate():.1%}")
    print(f"\n  Estimated tok/s for {config.num_layers} layers: "
          f"{1e6 / (np.mean(timings['total_us']) * config.num_layers):.2f}")


if __name__ == '__main__':
    main()
