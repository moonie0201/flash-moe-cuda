#!/usr/bin/env python3
"""35B end-to-end inference: non-expert on GPU, experts streamed from packed_experts_bf16.

Strategy:
  1. Instantiate model on meta device (no weights loaded)
  2. Load ONLY non-expert weights (~5-10GB) from safetensors to GPU
  3. Replace expert tensors with lazy stubs; monkey-patch forward to load from SSD
  4. Generate tokens, measure tok/s

Usage:
    python ses/src/run_397b_ssd.py --prompt "What is MoE?" --tokens 5 --hot-pct 0.3
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from packed_loader import PackedExpertLoader
from gptq_expert_loader import GPTQExpertLoader
from packed_gptq_loader import PackedGPTQLoader

MODEL_DIR = '/home/mh/models/Qwen3.5-35B-A3B'
PACKED_DIR = '/home/mh/models/Qwen3.5-35B-A3B/packed_experts_bf16'
GPTQ_DIR = ''  # not used for 35B (no GPTQ variant)
PACKED_GPTQ_DIR = ''


def mmap_safetensors(path):
    fd = os.open(path, os.O_RDONLY)
    size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
    hs = struct.unpack('<Q', mm[:8])[0]
    hdr = json.loads(mm[8:8+hs].decode('utf-8'))
    return mm, fd, 8 + hs, hdr


def load_tensor(mm, data_off, hdr, key, device, dtype):
    meta = hdr[key]
    start, end = meta['data_offsets']
    raw = mm[data_off + start:data_off + end]
    if meta['dtype'] == 'BF16':
        u16 = np.frombuffer(raw, dtype=np.uint16).reshape(meta['shape']).copy()
        t = torch.from_numpy(u16).view(torch.bfloat16)
    elif meta['dtype'] == 'F32':
        arr = np.frombuffer(raw, dtype=np.float32).reshape(meta['shape']).copy()
        t = torch.from_numpy(arr)
    elif meta['dtype'] == 'I32':
        arr = np.frombuffer(raw, dtype=np.int32).reshape(meta['shape']).copy()
        t = torch.from_numpy(arr)
    else:
        raise ValueError(f"Unsupported dtype: {meta['dtype']}")
    if meta['dtype'] == 'I32':
        return t.to(device)
    return t.to(device=device, dtype=dtype)


class ShardManager:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
            self.index = json.load(f)
        self._shards = {}

    def _open(self, shard):
        if shard not in self._shards:
            self._shards[shard] = mmap_safetensors(os.path.join(self.model_dir, shard))
        return self._shards[shard]

    def load(self, key, device, dtype=torch.bfloat16):
        if key not in self.index['weight_map']:
            return None
        shard = self.index['weight_map'][key]
        mm, fd, do, hdr = self._open(shard)
        return load_tensor(mm, do, hdr, key, device, dtype)

    def has(self, key):
        return key in self.index['weight_map']

    def close(self):
        for mm, fd, _, _ in self._shards.values():
            mm.close()
            os.close(fd)
        self._shards.clear()


def load_non_expert_weights(model, shards, layer_device_fn, verbose=True):
    """Load weights for every parameter EXCEPT expert gate_up_proj / down_proj.

    Returns number of parameters loaded.
    """
    loaded = 0
    total = 0
    skipped_experts = 0
    missing = []

    def set_submodule_param(mod, dotted, tensor):
        parts = dotted.split('.')
        for p in parts[:-1]:
            mod = getattr(mod, p)
        leaf = parts[-1]
        old = getattr(mod, leaf)
        if isinstance(old, torch.nn.Parameter):
            new_param = torch.nn.Parameter(tensor, requires_grad=False)
            setattr(mod, leaf, new_param)
        else:
            # buffer
            mod.register_buffer(leaf, tensor, persistent=False)

    t0 = time.time()
    for name, param in list(model.named_parameters()):
        total += 1
        if '.mlp.experts.' in name and ('gate_up_proj' in name or 'down_proj' in name):
            skipped_experts += 1
            continue

        # Name mapping: param "model.layers.X.Y" -> safetensors "model.language_model.layers.X.Y"
        candidates = [name]
        if name.startswith('model.') and not name.startswith('model.language_model.'):
            candidates.append('model.language_model.' + name[len('model.'):])
        key = None
        for c in candidates:
            if shards.has(c):
                key = c
                break
        if key is None:
            missing.append(name)
            continue

        device = torch.device('cuda:0')
        if '.layers.' in name:
            try:
                idx = int(name.split('.layers.')[1].split('.')[0])
                device = layer_device_fn(idx)
            except Exception:
                pass

        dtype = torch.bfloat16
        with torch.no_grad():
            tensor = shards.load(key, device, dtype=dtype)
            if tensor is None:
                continue
            set_submodule_param(model, name, tensor)
        loaded += 1

    elapsed = time.time() - t0
    if verbose:
        print(f"  Non-expert: {loaded}/{total} loaded ({skipped_experts} experts skipped), "
              f"{elapsed:.0f}s")
        if missing:
            print(f"  Missing keys: {len(missing)}, first few:")
            for m in missing[:5]:
                print(f"    {m}")
    return loaded


def attach_layer_device_hooks(model, layer_device_fn):
    """Move inputs to each decoder layer's device before its forward."""
    def make_hook(dev):
        def _move(x):
            if isinstance(x, torch.Tensor):
                return x.to(dev) if x.device != dev else x
            if isinstance(x, (tuple, list)):
                moved = type(x)(_move(v) for v in x)
                return moved
            if isinstance(x, dict):
                return {k: _move(v) for k, v in x.items()}
            return x
        def hook(module, args, kwargs):
            return tuple(_move(a) for a in args), {k: _move(v) for k, v in kwargs.items()}
        return hook

    layers = model.model.layers if hasattr(model.model, 'layers') else model.model.language_model.layers
    for i, layer in enumerate(layers):
        dev = layer_device_fn(i)
        layer.register_forward_pre_hook(make_hook(dev), with_kwargs=True)

    # After last layer, move back to cuda:0 for norm + lm_head
    last_dev = layer_device_fn(len(layers) - 1)
    final_dev = torch.device('cuda:0')
    if last_dev != final_dev:
        if hasattr(model.model, 'norm'):
            model.model.norm.register_forward_pre_hook(make_hook(final_dev), with_kwargs=True)


def _expert_swiglu(current, gu, dw, weights):
    """Single expert SwiGLU computation. Pure tensor ops for torch.compile compat."""
    gate_up = F.linear(current, gu)
    gate, up = gate_up.chunk(2, dim=-1)
    h = F.silu(gate) * up
    out = F.linear(h, dw)
    return out * weights.unsqueeze(-1)


class DecodeGraphRunner:
    """Captures a CUDA Graph for batch=1 SwiGLU expert compute and replays it.

    Decode mode has fixed shape (1 token at a time), making it an ideal CUDA Graph target.
    Prefill (variable batch) falls through to eager.
    """

    def __init__(self, hidden_size, intermediate_size, device, dtype=torch.bfloat16):
        self.hidden_size = hidden_size
        self.inter_size = intermediate_size
        self.device = device
        self.dtype = dtype

        # Static buffers (graph captures these specific tensors)
        self.in_x      = torch.zeros(1, hidden_size, device=device, dtype=dtype)
        self.in_gu     = torch.zeros(2 * intermediate_size, hidden_size, device=device, dtype=dtype)
        self.in_dw     = torch.zeros(hidden_size, intermediate_size, device=device, dtype=dtype)
        self.in_w      = torch.zeros(1, device=device, dtype=dtype)
        self.out_buf   = torch.zeros(1, hidden_size, device=device, dtype=dtype)

        self.graph = None

    def capture(self):
        """Warmup then capture graph."""
        # Warmup
        s = torch.cuda.Stream(self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            for _ in range(3):
                self.out_buf.copy_(_expert_swiglu(self.in_x, self.in_gu, self.in_dw, self.in_w))
        torch.cuda.current_stream(self.device).wait_stream(s)

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            self.out_buf.copy_(_expert_swiglu(self.in_x, self.in_gu, self.in_dw, self.in_w))
        self.graph = g

    def run(self, x, gu, dw, weights):
        """Run captured graph if shape matches; else fall through to eager."""
        if self.graph is None or x.shape[0] != 1:
            return _expert_swiglu(x, gu, dw, weights)
        # Copy inputs into static buffers
        self.in_x.copy_(x, non_blocking=True)
        self.in_gu.copy_(gu, non_blocking=True)
        self.in_dw.copy_(dw, non_blocking=True)
        self.in_w.copy_(weights, non_blocking=True)
        self.graph.replay()
        return self.out_buf.clone()


def patch_experts(model, packed_loader, hot_cache, gpu_cache=None, use_compile=False,
                  use_cuda_graph=False, use_batched_decode=False,
                  use_stream_overlap=False):
    """Replace Qwen3_5MoeExperts.forward with SSD-streaming version."""

    if gpu_cache is None:
        gpu_cache = {}

    stats = {'load_hit_s': 0.0, 'load_miss_s': 0.0, 'compute_s': 0.0,
             'hits': 0, 'misses': 0, 'gpu_hits': 0}
    activation_trace = {}  # (layer, eid) -> count
    packed_loader._ssd_stats = stats
    packed_loader._trace = activation_trace

    # Optional: compile the per-expert compute kernel
    if use_compile:
        compiled_swiglu = torch.compile(_expert_swiglu, mode='reduce-overhead',
                                         dynamic=True, fullgraph=False)
        print("  torch.compile applied to expert_swiglu (mode=reduce-overhead)")
    else:
        compiled_swiglu = _expert_swiglu

    # Optional: per-device decode-mode CUDA Graph runner
    decode_runners = {}
    if use_cuda_graph:
        # Get hidden/inter sizes from the model config
        cfg = next(iter(model.modules())).config if hasattr(model, 'config') else None
        # Fallback: detect from first MoE expert module
        for m in model.modules():
            if type(m).__name__ == 'Qwen3_5MoeExperts':
                hidden_size = m.hidden_size if hasattr(m, 'hidden_size') else 2048
                inter_size = m.moe_intermediate_size if hasattr(m, 'moe_intermediate_size') else 512
                break
        for dev in [torch.device('cuda:0')]:
            r = DecodeGraphRunner(hidden_size, inter_size, dev)
            r.capture()
            decode_runners[dev] = r
        print(f"  CUDA Graph captured: decode-mode batch=1 SwiGLU "
              f"(hidden={hidden_size}, inter={inter_size})")

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=16)

    is_gptq = isinstance(packed_loader, (GPTQExpertLoader, PackedGPTQLoader))
    is_packed_gptq = isinstance(packed_loader, PackedGPTQLoader)

    # One prefetch stream per GPU device
    prefetch_streams = {}
    def get_prefetch_stream(dev):
        if dev not in prefetch_streams:
            prefetch_streams[dev] = torch.cuda.Stream(dev)
        return prefetch_streams[dev]

    def _materialize_gpu(payload, is_hit, cache_fmt, device):
        """Materialize weights on GPU. Returns (gu, dw) bf16 [out, in]."""
        if is_hit == 'gpu':
            return payload  # already on VRAM, zero-copy
        if is_hit:
            if is_gptq and cache_fmt == 'raw':
                if is_packed_gptq:
                    return packed_loader.dequant_raw_bytes_to_gpu(payload, device)
                return packed_loader.dequant_raw_to_gpu(payload, device)
            # Combined buffer format: ('combined', pinned, gu_shape, dw_shape, gu_n)
            if isinstance(payload, tuple) and len(payload) == 5 and payload[0] == 'combined':
                _, pinned, gu_shape, dw_shape, gu_n = payload
                buf_gpu = pinned.to(device, non_blocking=True)
                gu = buf_gpu[:gu_n].view(gu_shape)
                dw = buf_gpu[gu_n:].view(dw_shape)
                return gu, dw
            gu_c, dw_c = payload
            return (gu_c.to(device, non_blocking=True),
                    dw_c.to(device, non_blocking=True))
        else:
            if is_gptq:
                if is_packed_gptq:
                    return packed_loader.dequant_raw_bytes_to_gpu(payload, device)
                return packed_loader.dequant_raw_to_gpu(payload, device)
            gu_c, dw_c = payload
            return (gu_c.to(device, non_blocking=True),
                    dw_c.to(device, non_blocking=True))

    def make_forward(layer_idx):
        def ssd_forward(self, hidden_states, top_k_index, top_k_weights):
            device = hidden_states.device
            dtype = hidden_states.dtype
            final = torch.zeros_like(hidden_states)

            unique_ids = torch.unique(top_k_index).tolist()
            if self.num_experts in unique_ids:
                unique_ids.remove(self.num_experts)

            for eid in unique_ids:
                k = (layer_idx, eid)
                activation_trace[k] = activation_trace.get(k, 0) + 1

            cache_fmt = hot_cache.get('__format__', 'bf16')
            t_load = time.perf_counter()

            # ===== Stage 3: Decode-mode batched path (single token) =====
            # When batch_size=1, all K experts process the same single token.
            # Stack their weights and do batched bmm in one GPU call.
            if (use_batched_decode and hidden_states.shape[0] == 1
                and not is_gptq):
                k_ids = top_k_index[0].tolist()  # K active expert IDs for this token
                k_weights = top_k_weights[0]     # [K]

                # Load all K weights from cache (or SSD via futures)
                gus, dws = [], []
                for eid in k_ids:
                    if eid == self.num_experts:
                        gus.append(None); dws.append(None); continue
                    if (layer_idx, eid) in hot_cache:
                        gu_c, dw_c = hot_cache[(layer_idx, eid)]
                        stats['hits'] += 1
                    else:
                        gu_c, dw_c = packed_loader.load_expert(layer_idx, eid, torch.device('cpu'))
                        stats['misses'] += 1
                    gus.append(gu_c)
                    dws.append(dw_c)

                # Filter out None slots (drop tokens)
                valid = [(i, g, d) for i, (g, d) in enumerate(zip(gus, dws)) if g is not None]
                if not valid:
                    return final

                # Stack and transfer in single op
                gu_stack = torch.stack([g for _, g, _ in valid]).to(device, non_blocking=True)
                dw_stack = torch.stack([d for _, _, d in valid]).to(device, non_blocking=True)
                w_active = torch.stack([k_weights[i] for i, _, _ in valid])  # [K_valid]

                # Batched compute (K, 1, hidden)
                K_valid = gu_stack.shape[0]
                x_b = hidden_states.unsqueeze(0).expand(K_valid, -1, -1)  # [K, 1, hidden]
                gate_up = torch.bmm(x_b, gu_stack.transpose(1, 2))  # [K, 1, 2*inter]
                gate, up = gate_up.chunk(2, dim=-1)
                h = F.silu(gate) * up  # [K, 1, inter]
                out = torch.bmm(h, dw_stack.transpose(1, 2))  # [K, 1, hidden]
                # Weighted sum
                out_weighted = (out.squeeze(1) * w_active.unsqueeze(-1)).sum(0, keepdim=True)
                final = final + out_weighted.to(dtype)

                torch.cuda.synchronize(device)
                stats['load_miss_s'] += time.perf_counter() - t_load
                return final

            # ===== Original per-expert path (prefill or non-batched) =====
            def _load_raw(eid):
                k = (layer_idx, eid)
                if k in gpu_cache:
                    return eid, gpu_cache[k], 'gpu'
                if k in hot_cache:
                    return eid, hot_cache[k], True
                if is_packed_gptq:
                    raw = packed_loader.load_expert_raw_bytes(layer_idx, eid)
                    return eid, raw, False
                if is_gptq:
                    raw = packed_loader.load_expert_raw(layer_idx, eid)
                    return eid, raw, False
                return eid, packed_loader.load_expert(layer_idx, eid, torch.device('cpu')), False

            futures = {eid: pool.submit(_load_raw, eid) for eid in unique_ids}

            # Stream-overlap path: prefetch expert N+1 on prefetch_stream
            # while computing expert N on compute_stream. Pure-transfer prefetch
            # truly overlaps with compute (different GPU engines).
            if use_stream_overlap and not is_gptq and unique_ids:
                compute_stream = torch.cuda.current_stream(device)
                prefetch_stream = get_prefetch_stream(device)

                # Issue first prefetch
                _, payload0, is_hit0 = futures[unique_ids[0]].result()
                with torch.cuda.stream(prefetch_stream):
                    next_gu, next_dw = _materialize_gpu(payload0, is_hit0, cache_fmt, device)
                    next_event = torch.cuda.Event()
                    next_event.record(prefetch_stream)
                next_is_hit = is_hit0

                for i, eid in enumerate(unique_ids):
                    cur_gu, cur_dw = next_gu, next_dw
                    cur_event = next_event
                    is_hit = next_is_hit

                    # Issue next prefetch (overlaps with current expert compute)
                    if i + 1 < len(unique_ids):
                        nid = unique_ids[i + 1]
                        _, np_, nh_ = futures[nid].result()
                        with torch.cuda.stream(prefetch_stream):
                            next_gu, next_dw = _materialize_gpu(np_, nh_, cache_fmt, device)
                            next_event = torch.cuda.Event()
                            next_event.record(prefetch_stream)
                        next_is_hit = nh_

                    # Wait only for current expert's transfer
                    compute_stream.wait_event(cur_event)
                    if is_hit == 'gpu': stats['gpu_hits'] += 1
                    elif is_hit:        stats['hits'] += 1
                    else:               stats['misses'] += 1

                    mask = (top_k_index == eid)
                    if not mask.any():
                        del cur_gu, cur_dw
                        continue
                    token_idx, k_pos = mask.nonzero(as_tuple=True)
                    current = hidden_states[token_idx]
                    weights = top_k_weights[token_idx, k_pos]
                    out = _expert_swiglu(current, cur_gu, cur_dw, weights)
                    final.index_add_(0, token_idx, out.to(dtype))
                    del cur_gu, cur_dw

                torch.cuda.synchronize(device)
                stats['load_miss_s'] += time.perf_counter() - t_load
                return final

            # Fallback: serial per-expert
            for i, eid in enumerate(unique_ids):
                _, payload, is_hit = futures[eid].result()
                gu, dw = _materialize_gpu(payload, is_hit, cache_fmt, device)

                if is_hit == 'gpu': stats['gpu_hits'] += 1
                elif is_hit:        stats['hits'] += 1
                else:               stats['misses'] += 1

                mask = (top_k_index == eid)
                if not mask.any():
                    del gu, dw
                    continue
                token_idx, k_pos = mask.nonzero(as_tuple=True)
                current = hidden_states[token_idx]

                weights = top_k_weights[token_idx, k_pos]
                runner = decode_runners.get(device)
                if runner is not None and current.shape[0] == 1:
                    out = runner.run(current, gu, dw, weights)
                else:
                    out = compiled_swiglu(current, gu, dw, weights)
                final.index_add_(0, token_idx, out.to(dtype))

                del gu, dw

            torch.cuda.synchronize(device)
            total_dt = time.perf_counter() - t_load
            stats['load_miss_s'] += total_dt

            return final
        return ssd_forward

    patched = 0
    for name, module in model.named_modules():
        if type(module).__name__ == 'Qwen3_5MoeExperts':
            # Extract layer index
            for i, p in enumerate(name.split('.')):
                if p == 'layers' and i + 1 < len(name.split('.')):
                    layer_idx = int(name.split('.')[i + 1])
                    break
            else:
                continue

            # Drop expert weights from the module (they were never loaded)
            for attr in ['gate_up_proj', 'down_proj']:
                if hasattr(module, attr):
                    delattr(module, attr)
                    module.register_buffer(attr, None, persistent=False)

            # Bind new forward
            import types as _types
            module.forward = _types.MethodType(make_forward(layer_idx), module)
            patched += 1
    return patched


def build_full_cache(num_layers, num_experts, packed_loader, combined_buf=False):
    """Cache ALL experts in pinned CPU memory. Eliminates all SSD misses.

    For 35B: 40 × 256 × 6MB = ~60GB pinned RAM.

    combined_buf: if True, store gu+dw in single contiguous pinned tensor per expert.
                  Halves PCIe transfer count (one big DMA instead of two).
    """
    print(f"Building FULL cache: {num_layers * num_experts} experts "
          f"({'combined buf' if combined_buf else 'separate'})...")
    cache = {}
    t0 = time.time()

    def _pin(t):
        try:
            return t.pin_memory()
        except RuntimeError:
            return t

    for layer in range(num_layers):
        for eid in range(num_experts):
            gu, dw = packed_loader.load_expert(layer, eid, torch.device('cpu'))
            if combined_buf:
                # Concatenate gu (flat) + dw (flat) into one buffer
                gu_n = gu.numel()
                dw_n = dw.numel()
                combined = torch.empty(gu_n + dw_n, dtype=gu.dtype)
                combined[:gu_n] = gu.reshape(-1)
                combined[gu_n:] = dw.reshape(-1)
                pinned = _pin(combined)
                # Store metadata for reshape on GPU side
                cache[(layer, eid)] = ('combined', pinned, gu.shape, dw.shape, gu_n)
            else:
                cache[(layer, eid)] = (_pin(gu), _pin(dw))
        if (layer + 1) % 5 == 0:
            if combined_buf:
                mb = sum(p[1].nelement()*2 for p in cache.values())/1e9
            else:
                mb = sum((gu.nelement()*2 + dw.nelement()*2) for (gu, dw) in cache.values())/1e9
            print(f"  layer {layer+1}/{num_layers}: {len(cache)} experts, {mb:.1f}GB, "
                  f"{time.time()-t0:.0f}s", flush=True)

    print(f"FULL cache: {len(cache)} experts, {time.time()-t0:.0f}s")
    return cache, 'bf16'


def build_hot_cache_from_trace(activation_trace, packed_loader, top_k_experts,
                                cache_format='auto'):
    """Build HOT cache from real activation frequencies.

    cache_format:
      - 'bf16': store dequanted bf16 weights on CPU pinned (fast hit, 24MB/expert)
      - 'raw':  store raw GPTQ tensors on CPU pinned (6.5MB/expert, dequant on each hit)
      - 'auto': bf16 if RAM allows all experts, else raw (GPTQ only)
    """
    ranked = sorted(activation_trace.items(), key=lambda x: -x[1])[:top_k_experts]
    is_gptq = isinstance(packed_loader, (GPTQExpertLoader, PackedGPTQLoader))
    is_packed_gptq = isinstance(packed_loader, PackedGPTQLoader)

    if not is_gptq:
        cache_format = 'bf16'
    elif cache_format == 'auto':
        # For GPTQ: prefer bf16 if it fits (~24MB × N ≤ 100GB → N ≤ 4300)
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
        if len(ranked) * 24 / 1024 < avail_gb * 0.75:
            cache_format = 'bf16'
        else:
            cache_format = 'raw'

    per_mb = 24.0 if cache_format == 'bf16' else 6.5
    print(f"Building HOT cache: top {len(ranked)} experts "
          f"(format={cache_format}, ~{per_mb:.1f}MB each)...")
    cache = {}
    t0 = time.time()

    def _pin(t):
        try:
            return t.pin_memory()
        except RuntimeError:
            return t

    # For bf16 format: dequant on GPU once, move to CPU pinned (bf16)
    gpu_dev = torch.device('cuda:0')
    for (layer, eid), cnt in ranked:
        if cache_format == 'bf16':
            if is_gptq:
                gu, dw = packed_loader.load_expert(layer, eid, gpu_dev)  # dequant on GPU
                gu_cpu = gu.cpu()
                dw_cpu = dw.cpu()
                del gu, dw
                cache[(layer, eid)] = (_pin(gu_cpu), _pin(dw_cpu))
            else:
                gu, dw = packed_loader.load_expert(layer, eid, torch.device('cpu'))
                cache[(layer, eid)] = (_pin(gu), _pin(dw))
        else:  # 'raw' — GPTQ only
            if is_packed_gptq:
                raw = packed_loader.load_expert_raw_bytes(layer, eid)
                # raw is a numpy uint8 array; wrap in torch tensor and pin
                t = torch.from_numpy(raw)
                cache[(layer, eid)] = _pin(t)
            else:
                raw = packed_loader.load_expert_raw(layer, eid)
                pinned_raw = tuple(tuple(_pin(t) for t in proj) for proj in raw)
                cache[(layer, eid)] = pinned_raw

    gb = len(cache) * per_mb / 1024
    print(f"HOT cache: {len(cache)} experts, ~{gb:.1f}GB, {time.time()-t0:.0f}s, "
          f"format={cache_format}")
    return cache, cache_format


def build_gpu_hot_cache(activation_trace, packed_loader, layer_device_fn,
                        max_vram_gb=4.0, safety_margin_gb=1.5):
    """Load top-N experts directly into VRAM. Hit = zero PCIe transfer.

    Respects per-device free VRAM with safety_margin_gb headroom.
    max_vram_gb caps the total budget across all devices.
    """
    ranked = sorted(activation_trace.items(), key=lambda x: -x[1])
    expert_mb = 6.0  # bf16 gate_up(4MB) + down(2MB)

    # Per-device available budget (free VRAM minus safety margin)
    n_dev = torch.cuda.device_count()
    dev_budget_mb = {}
    for i in range(n_dev):
        free_gb = torch.cuda.mem_get_info(i)[0] / 1e9
        dev_budget_mb[i] = max(0.0, free_gb - safety_margin_gb) * 1024
    total_budget_mb = min(max_vram_gb * 1024, sum(dev_budget_mb.values()))

    print(f"Building GPU HOT cache (≤{max_vram_gb:.1f}GB total, {expert_mb:.0f}MB/expert)...")
    for i in range(n_dev):
        print(f"  GPU {i}: {dev_budget_mb[i]/1024:.1f}GB available for cache")

    cache = {}
    used_mb = {i: 0.0 for i in range(n_dev)}
    total_used_mb = 0.0
    t0 = time.time()

    oom_devices = set()
    for (layer, eid), cnt in ranked:
        if total_used_mb + expert_mb > total_budget_mb:
            break
        device = layer_device_fn(layer)
        dev_idx = device.index if device.index is not None else 0
        if dev_idx in oom_devices:
            continue
        if used_mb[dev_idx] + expert_mb > dev_budget_mb[dev_idx]:
            continue  # this device is full, skip (may still load to other devices)
        try:
            gu, dw = packed_loader.load_expert(layer, eid, device)
        except torch.cuda.OutOfMemoryError:
            oom_devices.add(dev_idx)
            torch.cuda.empty_cache()
            continue
        cache[(layer, eid)] = (gu, dw)
        used_mb[dev_idx] += expert_mb
        total_used_mb += expert_mb

    gb = total_used_mb / 1024
    print(f"GPU HOT cache: {len(cache)} experts, {gb:.1f}GB total")
    for i in range(n_dev):
        print(f"  GPU {i}: {torch.cuda.memory_allocated(i)/1e9:.2f}GB allocated "
              f"(+{used_mb[i]/1024:.1f}GB cache)")
    return cache


def build_hot_cache(shards, hot_pct, num_layers, num_experts, num_experts_per_tok,
                    hidden_size, packed_loader, profile_samples=200):
    """Profile expert activation with random inputs → cache top experts on pinned CPU memory."""
    print(f"Building HOT cache (top {hot_pct:.0%}, {profile_samples} profile samples)...")
    np.random.seed(42)
    n_hot = int(num_experts * hot_pct)
    cache = {}

    t0 = time.time()
    for layer in range(num_layers):
        # Load gate weight from safetensors
        gate_key = f'model.language_model.layers.{layer}.mlp.gate.weight'
        if not shards.has(gate_key):
            continue
        gate_w = shards.load(gate_key, 'cuda:0', dtype=torch.bfloat16)

        # Profile activation frequency
        freq = torch.zeros(num_experts, device='cuda:0')
        with torch.no_grad():
            for _ in range(profile_samples):
                x = torch.randn(hidden_size, device='cuda:0', dtype=torch.bfloat16)
                s = F.linear(x, gate_w).float()
                p = F.softmax(s, dim=0)
                ids = torch.topk(p, num_experts_per_tok).indices
                freq[ids] += 1

        hot_ids = torch.argsort(freq, descending=True)[:n_hot].cpu().tolist()
        del gate_w

        # Load those experts to pinned CPU memory
        for eid in hot_ids:
            gu, dw = packed_loader.load_expert(layer, eid, torch.device('cpu'))
            cache[(layer, eid)] = (gu.pin_memory(), dw.pin_memory())

        if (layer + 1) % 10 == 0:
            gb = len(cache) * 24 / 1024
            print(f"  layer {layer+1}/{num_layers}: {len(cache)} experts, "
                  f"{gb:.1f}GB, {time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    gb = len(cache) * 24 / 1024
    print(f"HOT cache: {len(cache)} experts, {gb:.1f}GB, {elapsed:.0f}s")
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='What is a Mixture of Experts model?')
    parser.add_argument('--tokens', type=int, default=5)
    parser.add_argument('--hot-pct', type=float, default=0.0,
                        help='Fraction of experts to HOT-cache (0=SSD only)')
    parser.add_argument('--max-layers', type=int, default=None,
                        help='Only use first N layers (debug)')
    parser.add_argument('--gptq', action='store_true',
                        help='Use GPTQ-Int4 experts (4x smaller, on-GPU dequant)')
    parser.add_argument('--cache-format', default='auto',
                        choices=['auto', 'bf16', 'raw'],
                        help='HOT cache format (bf16=dequanted 24MB/exp, raw=GPTQ 6.5MB/exp)')
    parser.add_argument('--full-cache', action='store_true',
                        help='Cache ALL experts in pinned RAM (eliminates SSD misses)')
    parser.add_argument('--torch-compile', action='store_true',
                        help='Wrap expert_swiglu with torch.compile(reduce-overhead)')
    parser.add_argument('--cuda-graph', action='store_true',
                        help='Capture decode-mode SwiGLU as CUDA Graph (batch=1)')
    parser.add_argument('--batched-decode', action='store_true',
                        help='Stage 3: batched K-expert bmm in single kernel for decode mode')
    parser.add_argument('--stream-overlap', action='store_true',
                        help='Double-buffered: prefetch expert N+1 while compute N')
    parser.add_argument('--combined-buf', action='store_true',
                        help='Combine gu+dw into single pinned buffer (1 transfer per expert)')
    parser.add_argument('--gpu-hot-gb', type=float, default=0.0,
                        help='VRAM budget for GPU-resident expert cache in GB (0=disabled)')
    parser.add_argument('--dual-gpu', action='store_true',
                        help='Split layers across GPU 0 and GPU 1 (doubles GPU cache budget)')
    parser.add_argument('--trace-tokens', type=int, default=10,
                        help='Tokens to generate for cold trace (more = better GPU cache coverage)')
    args = parser.parse_args()

    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    print("1. Loading config + tokenizer...")
    config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    tc = config.text_config
    n_layers = args.max_layers or tc.num_hidden_layers
    print(f"  Model: {tc.num_hidden_layers} layers, {tc.num_experts} experts, "
          f"K={tc.num_experts_per_tok}, hidden={tc.hidden_size}")

    print("2. Instantiating model on meta device (no weights)...")
    t0 = time.time()
    # Use text_config directly — outer composite config lacks flat vocab_size
    from transformers import Qwen3_5MoeForCausalLM
    with torch.device('meta'):
        model = Qwen3_5MoeForCausalLM(config.text_config)
    model = model.to(torch.bfloat16)
    print(f"  Meta init: {time.time()-t0:.1f}s")

    # 35B non-expert is ~5-7GB; fits on single GPU, leaves room for expert streaming
    n_gpus = torch.cuda.device_count()
    if args.dual_gpu and n_gpus >= 2:
        split = tc.num_hidden_layers // 2
        def layer_device(idx):
            return torch.device('cuda:0' if idx < split else 'cuda:1')
        print(f"  Dual-GPU: layers 0-{split-1} → GPU 0, {split}-{tc.num_hidden_layers-1} → GPU 1")
    else:
        def layer_device(idx):
            return torch.device('cuda:0')

    print("3. Loading non-expert weights...")
    shards = ShardManager(MODEL_DIR)
    load_non_expert_weights(model, shards, layer_device, verbose=True)
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.memory_allocated(i)/1e9:.2f}GB")

    print("4. Attaching layer device hooks...")
    attach_layer_device_hooks(model, layer_device)

    if args.gptq:
        # NOTE: PackedGPTQLoader produced garbage output under memory pressure
        # (likely pin_memory / byte-alignment / GPU view lifetime issue).
        # GPTQExpertLoader (safetensors scatter) is the verified correct path.
        print("5. Setting up GPTQExpertLoader (safetensors scatter)...")
        packed_loader = GPTQExpertLoader(GPTQ_DIR, device=torch.device('cuda:0'))
    else:
        print("5. Setting up PackedExpertLoader (bf16)...")
        packed_loader = PackedExpertLoader(PACKED_DIR, device=torch.device('cuda:0'))

    # HOT cache starts empty; populated after COLD trace pass
    hot_cache = {}

    print("6. Patching expert forward with SSD streaming...")
    gpu_cache = {}
    n_patched = patch_experts(model, packed_loader, hot_cache, gpu_cache=gpu_cache,
                              use_compile=args.torch_compile,
                              use_cuda_graph=args.cuda_graph,
                              use_batched_decode=args.batched_decode,
                              use_stream_overlap=args.stream_overlap)
    print(f"  Patched {n_patched} MoE layers")

    print("7. Generating...")
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
    tok_out = tokenizer(text, return_tensors="pt")
    input_ids = tok_out.input_ids.to('cuda:0')
    attention_mask = tok_out.attention_mask.to('cuda:0')

    model.eval()
    s = getattr(packed_loader, '_ssd_stats', None)

    def run_once(label, max_tokens=None):
        if s:
            for k in s: s[k] = 0 if isinstance(s[k], int) else 0.0
        n = max_tokens if max_tokens is not None else args.tokens
        t = time.time()
        with torch.no_grad():
            out = model.generate(input_ids, attention_mask=attention_mask,
                                 max_new_tokens=n, do_sample=False,
                                 use_cache=True)
        el = time.time() - t
        nt = out.shape[1] - input_ids.shape[1]
        resp = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n=== [{label}] {nt} tokens in {el:.1f}s ({nt/el:.3f} tok/s) ===")
        print(f"Output: {resp}")
        if s:
            total = s['gpu_hits'] + s['hits'] + s['misses']
            print(f"Breakdown: gpu_hit={s['gpu_hits']}×, "
                  f"cpu_hit={s['hits']}× ({s['load_hit_s']:.1f}s), "
                  f"miss={s['misses']}× ({s['load_miss_s']:.1f}s), "
                  f"compute={s['compute_s']:.1f}s")
            if total:
                print(f"  gpu_hit_rate={s['gpu_hits']/total:.1%}  "
                      f"cpu_hit_rate={s['hits']/total:.1%}  "
                      f"miss_rate={s['misses']/total:.1%}")
            if s['misses']:
                print(f"  avg miss: {s['load_miss_s']/s['misses']*1000:.1f}ms/expert")
            if s['hits']:
                print(f"  avg cpu_hit: {s['load_hit_s']/s['hits']*1000:.1f}ms/expert")

    if args.full_cache:
        # If GPU cache requested, do a short trace pass to identify hot experts
        if args.gpu_hot_gb > 0:
            run_once("COLD (trace for GPU cache)", max_tokens=args.trace_tokens)
            torch.cuda.empty_cache()  # release KV cache before VRAM budget query
            trace = getattr(packed_loader, '_trace', {})
            print(f"Traced {len(trace)} unique (layer, expert) activations")
            built_gpu = build_gpu_hot_cache(trace, packed_loader, layer_device,
                                            max_vram_gb=args.gpu_hot_gb)
            gpu_cache.update(built_gpu)
            gpu_covered = sum(cnt for k, cnt in trace.items() if k in gpu_cache)
            total_obs = sum(trace.values())
            print(f"GPU cache covers {gpu_covered}/{total_obs} = {gpu_covered/total_obs:.1%}")

        # Cache ALL remaining experts in CPU pinned RAM
        built, cache_fmt = build_full_cache(tc.num_hidden_layers, tc.num_experts,
                                              packed_loader, combined_buf=args.combined_buf)
        hot_cache.update(built)
        hot_cache['__format__'] = cache_fmt
        label = "FULL+GPU" if args.gpu_hot_gb > 0 else "FULL"
        run_once(label)
    else:
        # Pass 1: COLD, traces activations
        run_once("COLD (trace)")
        trace = getattr(packed_loader, '_trace', {})
        print(f"Traced {len(trace)} unique (layer, expert) activations")

        # Build HOT cache from trace if requested
        if args.hot_pct > 0:
            total_slots = int(tc.num_hidden_layers * tc.num_experts * args.hot_pct)
            built, cache_fmt = build_hot_cache_from_trace(trace, packed_loader, total_slots,
                                                           cache_format=args.cache_format)
            hot_cache.update(built)
            hot_cache['__format__'] = cache_fmt
            covered = sum(cnt for k, cnt in trace.items() if k in hot_cache)
            total_obs = sum(trace.values())
            print(f"HOT cache covers {covered}/{total_obs} = {covered/total_obs:.1%}")

        # Build GPU-resident cache from trace (top experts by frequency into VRAM)
        if args.gpu_hot_gb > 0:
            built_gpu = build_gpu_hot_cache(trace, packed_loader, layer_device,
                                            max_vram_gb=args.gpu_hot_gb)
            gpu_cache.update(built_gpu)
            gpu_covered = sum(cnt for k, cnt in trace.items() if k in gpu_cache)
            total_obs = sum(trace.values())
            print(f"GPU cache covers {gpu_covered}/{total_obs} = {gpu_covered/total_obs:.1%}")

        # Pass 2: with HOT + GPU cache
        run_once("HOT+GPU" if args.gpu_hot_gb > 0 else "HOT")

    # Cleanup
    shards.close()
    if hasattr(packed_loader, 'close'):
        packed_loader.close()


if __name__ == '__main__':
    main()
