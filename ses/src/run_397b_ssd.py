#!/usr/bin/env python3
"""397B end-to-end inference: non-expert on GPU, experts streamed from packed_experts_bf16.

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
from two_bit_loader import TwoBitLoader

MODEL_DIR = '/home/mh/models/Qwen3.5-397B-A17B'
PACKED_DIR = os.path.join(MODEL_DIR, 'packed_experts_bf16')
PACKED_2BIT_DIR = os.path.join(MODEL_DIR, 'packed_experts_2bit')
GPTQ_DIR  = '/home/mh/models/Qwen3.5-397B-A17B-GPTQ-Int4'
PACKED_GPTQ_DIR = os.path.join(GPTQ_DIR, 'packed_experts')


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


def patch_experts(model, packed_loader, hot_cache, gpu_cache=None):
    """Replace Qwen3_5MoeExperts.forward with SSD-streaming version."""

    if gpu_cache is None:
        gpu_cache = {}
    stats = {'load_hit_s': 0.0, 'load_miss_s': 0.0, 'compute_s': 0.0,
             'hits': 0, 'gpu_hits': 0, 'misses': 0}
    activation_trace = {}  # (layer, eid) -> count
    # Deferred CPU tensor lifetime: [(cuda_event, cpu_refs_list), ...]
    # Avoids per-layer synchronize — GPU stream ordering ensures correctness,
    # but we must keep source tensors alive until their H2D DMA completes.
    _pending_refs = []
    packed_loader._ssd_stats = stats
    packed_loader._trace = activation_trace

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=16)
    # Pre-allocated pinned buffers — reused every layer to avoid cudaMallocHost overhead
    _rw_buf = torch.zeros(20, dtype=torch.float32).pin_memory()   # routing weights [N]

    # Cross-layer prefetch state (mutable via single-element lists for closure mutation)
    _token_acts = []          # [{layer: frozenset(eids)}, ...] per token, COLD pass
    _cur_acts = [{}]          # current token's {layer: frozenset(eids)}
    _prefetch = [{}]          # {eid: gpu_tensor} ready on device for current layer
    _prefetch_cpu = [[]]      # cpu tensors keeping H2D source alive
    _cross_table = [{}]       # {(layer, eid): [top_eids_next_layer]} — built post-COLD
    _prefetch_enabled = [False]
    _prefetch_stats = [0, 0]  # [used_prefetch_hits, total_prefetch_started]
    _hs_trace = []            # [(layer_idx, h_normed_np, expert_ids_list), ...] COLD only
    _collect_hs = [False]     # enabled by --save-hs before COLD pass
    _predictor = [None]       # CrossLayerExpertPredictor instance (loaded post-COLD)
    packed_loader._token_acts = _token_acts
    packed_loader._cur_acts = _cur_acts
    packed_loader._prefetch = _prefetch
    packed_loader._cross_table = _cross_table
    packed_loader._prefetch_enabled = _prefetch_enabled
    packed_loader._prefetch_stats = _prefetch_stats
    packed_loader._hs_trace = _hs_trace
    packed_loader._collect_hs = _collect_hs
    packed_loader._predictor = _predictor

    is_2bit = isinstance(packed_loader, TwoBitLoader)
    is_gptq = isinstance(packed_loader, (GPTQExpertLoader, PackedGPTQLoader))
    is_packed_gptq = isinstance(packed_loader, PackedGPTQLoader)

    # Try to load the compiled CUDA extension for batched 2-bit expert forward.
    # Falls back to the serial Python path if not built yet.
    _expert_ops = None
    if is_2bit:
        try:
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import expert_ops as _expert_ops
            print("  [expert_ops] CUDA batched forward: ENABLED")
        except ImportError:
            print("  [expert_ops] CUDA extension not found — using serial Python path")
            print("    Build with: cd ses/src && python setup_expert_ops.py build_ext --inplace")

    def _materialize_gpu(payload, is_hit, cache_fmt, device):
        """Materialize weights on GPU. Returns (gu, dw) bf16 [out, in]."""
        if is_hit == 'gpu':
            return packed_loader.dequant_raw_to_gpu(payload, device)
        if is_hit:
            if is_2bit and cache_fmt == 'raw':
                return packed_loader.dequant_raw_to_gpu(payload, device)
            if is_gptq and cache_fmt == 'raw':
                if is_packed_gptq:
                    return packed_loader.dequant_raw_bytes_to_gpu(payload, device)
                return packed_loader.dequant_raw_to_gpu(payload, device)
            gu_c, dw_c = payload
            return (gu_c.to(device, non_blocking=True),
                    dw_c.to(device, non_blocking=True))
        else:
            if is_2bit:
                return packed_loader.dequant_raw_to_gpu(payload, device)
            if is_gptq:
                if is_packed_gptq:
                    return packed_loader.dequant_raw_bytes_to_gpu(payload, device)
                return packed_loader.dequant_raw_to_gpu(payload, device)
            gu_c, dw_c = payload
            return (gu_c.to(device, non_blocking=True),
                    dw_c.to(device, non_blocking=True))

    # 2-bit model shape constants (Qwen3.5-397B)
    _GU_ROWS, _GU_COLS = packed_loader.gu_shape if is_2bit else (None, None)
    _DN_ROWS, _DN_COLS = packed_loader.dn_shape if is_2bit else (None, None)
    _GROUP_SIZE = packed_loader.group_size if is_2bit else None

    def make_forward(layer_idx):
        def ssd_forward(self, hidden_states, top_k_index, top_k_weights):
            device = hidden_states.device
            dtype = hidden_states.dtype
            T = hidden_states.shape[0]

            unique_ids = torch.unique(top_k_index).tolist()
            if self.num_experts in unique_ids:
                unique_ids.remove(self.num_experts)
            N = len(unique_ids)

            for eid in unique_ids:
                k = (layer_idx, eid)
                activation_trace[k] = activation_trace.get(k, 0) + 1

            # ── ACTIVATION TRACKING (for cross-layer prefetch table building) ──
            fset = frozenset(unique_ids)
            if layer_idx == 0 and _cur_acts[0]:
                _token_acts.append(dict(_cur_acts[0]))
                _cur_acts[0].clear()
            _cur_acts[0][layer_idx] = fset

            # ── HIDDEN STATE COLLECTION (for predictor training, COLD pass only) ──
            if _collect_hs[0] and T == 1:
                _hs_trace.append((layer_idx,
                                  hidden_states[0].cpu().float().numpy(),
                                  list(unique_ids)))

            # ── BATCHED PATH (2-bit, single token, CUDA extension loaded) ──────
            if _expert_ops is not None and is_2bit and T == 1:
                # Release CPU tensors from previous layers whose H2D DMA is done
                _pending_refs[:] = [(e, r) for (e, r) in _pending_refs if not e.query()]
                t_load = time.perf_counter()

                # Consume prefetch from previous layer (H2D started during attention)
                avail_pf = _prefetch[0]
                _prefetch[0] = {}
                _prefetch_cpu[0] = []  # release previous cpu refs (H2D done by now)

                # Routing weights on CPU — computed BEFORE any GPU work is queued.
                # Line 267's .tolist() already flushed the stream; these D2H copies
                # are instant (no pending GPU work). GPU kernel launch overhead
                # dominates over actual compute at K=10 scale — CPU is faster.
                k_ids_cpu = top_k_index[0].tolist()    # K=10 ints, instant after sync
                k_wts_cpu = top_k_weights[0].tolist()  # K=10 floats, instant after sync
                uid_to_idx = {eid: i for i, eid in enumerate(unique_ids)}
                _rw_buf[:N].zero_()
                for eid, wt in zip(k_ids_cpu, k_wts_cpu):
                    idx = uid_to_idx.get(eid)
                    if idx is not None:
                        _rw_buf[idx] += wt
                routing_weights = _rw_buf[:N].to(device, non_blocking=True)

                # GPU/CPU cache: inline dict lookup (no thread pool overhead).
                # Only submit true SSD misses to thread pool for parallel I/O.
                # gpu_cache checked first — prefetch only helps cpu_hit experts.
                miss_futures = {}
                payload_map = {}
                for eid in unique_ids:
                    if (layer_idx, eid) in gpu_cache:
                        payload_map[eid] = (gpu_cache[(layer_idx, eid)], 'gpu')
                    elif eid in avail_pf:
                        payload_map[eid] = (avail_pf[eid], 'prefetch')  # H2D already done
                    elif (layer_idx, eid) in hot_cache:
                        payload_map[eid] = (hot_cache[(layer_idx, eid)], 'cpu')
                    else:
                        miss_futures[eid] = pool.submit(
                            packed_loader.load_expert_raw_bytes, layer_idx, eid)

                raw_list = []
                cpu_refs = []  # keep source tensors alive until H2D DMA completes
                for eid in unique_ids:
                    if eid in miss_futures:
                        cpu_t = torch.from_numpy(miss_futures[eid].result())
                        cpu_refs.append(cpu_t)
                        raw_list.append(cpu_t.to(device, non_blocking=True))
                        stats['misses'] += 1
                    else:
                        payload, src = payload_map[eid]
                        if src in ('gpu', 'prefetch'):
                            raw_list.append(payload)
                            stats['gpu_hits'] += 1
                            if src == 'prefetch':
                                _prefetch_stats[0] += 1
                        else:
                            cpu_t = payload if not isinstance(payload, np.ndarray) \
                                    else torch.from_numpy(payload)
                            cpu_refs.append(cpu_t)
                            raw_list.append(cpu_t.to(device, non_blocking=True))
                            stats['hits'] += 1

                # Stack all raw bytes: [N, expert_size] uint8 on GPU
                raw_batch = torch.stack(raw_list)  # [N, expert_size]

                # Single CUDA call: fused dequant + batched GEMV + SiLU + accumulate
                out = _expert_ops.expert_batch_forward_2bit(
                    raw_batch,
                    hidden_states[0],   # [gu_cols] bf16
                    routing_weights,
                    _GU_ROWS, _GU_COLS,
                    _DN_ROWS, _DN_COLS,
                    _GROUP_SIZE,
                )  # [dn_rows] bf16

                # Record event after all GPU work queued; defer cpu_refs release
                # until event completes (H2D DMA done). No CPU stall needed —
                # CUDA stream ordering guarantees the output is ready before
                # any downstream GPU op that reads it.
                if cpu_refs:
                    ev = torch.cuda.Event()
                    ev.record(torch.cuda.current_stream(device))
                    _pending_refs.append((ev, cpu_refs))
                stats['load_miss_s'] += time.perf_counter() - t_load

                # ── PRODUCE PREFETCH for layer+1 (start H2D during next attention) ──
                if _prefetch_enabled[0] and layer_idx < 59:
                    next_layer = layer_idx + 1
                    next_device = torch.device('cuda:0') if next_layer < 24 else torch.device('cuda:1')

                    # Predictor path: trained MLP predicts exactly top-K experts
                    pred_model = _predictor[0]
                    if pred_model is not None:
                        with torch.no_grad():
                            logits = pred_model(hidden_states[0].cpu().float(), layer_idx)
                        predicted_eids = logits.topk(10).indices.tolist()
                    else:
                        # Fallback: global top-K from n-gram table (max 10)
                        table = _cross_table[0]
                        scores = {}
                        for eid in unique_ids:
                            for rank, pred_eid in enumerate(table.get((layer_idx, eid), [])):
                                scores[pred_eid] = scores.get(pred_eid, 0) + (4 - rank)
                        predicted_eids = [e for e, _ in
                                          sorted(scores.items(), key=lambda x: -x[1])][:10]

                    new_pf = {}
                    new_pf_cpu = []
                    for next_eid in predicted_eids:
                        key = (next_layer, next_eid)
                        # Only prefetch cpu_hit experts (skip gpu_cache — already on GPU)
                        if key in hot_cache and key not in gpu_cache and next_eid not in new_pf:
                            cpu_t = hot_cache[key]
                            if isinstance(cpu_t, np.ndarray):
                                cpu_t = torch.from_numpy(cpu_t)
                            gpu_t = cpu_t.to(next_device, non_blocking=True)
                            new_pf[next_eid] = gpu_t
                            new_pf_cpu.append(cpu_t)
                            _prefetch_stats[1] += 1
                    _prefetch[0] = new_pf
                    _prefetch_cpu[0] = new_pf_cpu  # keep alive until next layer consumes

                return out.unsqueeze(0)  # [1, hidden_dim]

            # ── SERIAL FALLBACK PATH (non-2bit, multi-token, or extension missing) ─
            cache_fmt = hot_cache.get('__format__', 'bf16')
            final = torch.zeros_like(hidden_states)

            def _load_raw(eid):
                if (layer_idx, eid) in gpu_cache:
                    return eid, gpu_cache[(layer_idx, eid)], 'gpu'
                if (layer_idx, eid) in hot_cache:
                    return eid, hot_cache[(layer_idx, eid)], True
                if is_2bit or is_packed_gptq:
                    raw = packed_loader.load_expert_raw_bytes(layer_idx, eid)
                    return eid, raw, False
                if is_gptq:
                    raw = packed_loader.load_expert_raw(layer_idx, eid)
                    return eid, raw, False
                return eid, packed_loader.load_expert(layer_idx, eid, torch.device('cpu')), False

            t_load = time.perf_counter()
            futures = {eid: pool.submit(_load_raw, eid) for eid in unique_ids}

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

                gate_up = F.linear(current, gu)
                gate, up = gate_up.chunk(2, dim=-1)
                h = F.silu(gate) * up
                out = F.linear(h, dw)
                out = out * top_k_weights[token_idx, k_pos].unsqueeze(-1)
                final.index_add_(0, token_idx, out.to(dtype))

                del gu, dw

            torch.cuda.synchronize(device)
            stats['load_miss_s'] += time.perf_counter() - t_load
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


def build_hot_cache_from_trace(activation_trace, packed_loader, top_k_experts,
                                cache_format='auto'):
    """Build HOT cache from real activation frequencies.

    cache_format:
      - 'bf16': store dequanted bf16 weights on CPU pinned (fast hit, 24MB/expert)
      - 'raw':  store raw GPTQ tensors on CPU pinned (6.5MB/expert, dequant on each hit)
      - 'auto': bf16 if RAM allows all experts, else raw (GPTQ only)
    """
    ranked = sorted(activation_trace.items(), key=lambda x: -x[1])[:top_k_experts]
    is_2bit = isinstance(packed_loader, TwoBitLoader)
    is_gptq = isinstance(packed_loader, (GPTQExpertLoader, PackedGPTQLoader))
    is_packed_gptq = isinstance(packed_loader, PackedGPTQLoader)

    if is_2bit:
        cache_format = 'raw'  # store 3.38MB raw bytes, dequant on each hit
    elif not is_gptq:
        cache_format = 'bf16'
    elif cache_format == 'auto':
        # For GPTQ: prefer bf16 if it fits (~24MB × N ≤ 100GB → N ≤ 4300)
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
        if len(ranked) * 24 / 1024 < avail_gb * 0.75:
            cache_format = 'bf16'
        else:
            cache_format = 'raw'

    if is_2bit:
        per_mb = 3.38
    elif cache_format == 'bf16':
        per_mb = 24.0
    else:
        per_mb = 6.5
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
        else:  # 'raw' — GPTQ or 2-bit
            if is_2bit or is_packed_gptq:
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


def build_gpu_hot_cache_2bit(activation_trace, packed_loader, layer_device_fn,
                              max_vram_gb=10.0, safety_margin_gb=2.5):
    """Build GPU-resident raw 2-bit expert cache from activation trace.

    Stores 3.54MB uint8 tensors on GPU. Hit path: GPU dequant (no PCIe).
    Per-device budgeting with OOM exception handling.
    """
    ranked = sorted(activation_trace.items(), key=lambda x: -x[1])
    expert_mb = 3.54  # 2-bit expert size
    n_dev = torch.cuda.device_count()

    dev_budget_mb = {}
    for i in range(n_dev):
        free_gb = torch.cuda.mem_get_info(i)[0] / 1e9
        dev_budget_mb[i] = max(0.0, free_gb - safety_margin_gb) * 1024

    total_budget_mb = min(max_vram_gb * 1024, sum(dev_budget_mb.values()))
    print(f"Building GPU HOT cache (≤{max_vram_gb:.1f}GB total, {expert_mb:.1f}MB/expert)...")
    for i in range(n_dev):
        free_gb = torch.cuda.mem_get_info(i)[0] / 1e9
        print(f"  GPU {i}: {free_gb:.1f}GB available for cache")

    cache = {}
    used_mb = {i: 0.0 for i in range(n_dev)}
    total_used_mb = 0.0
    oom_devices = set()

    for (layer, eid), cnt in ranked:
        if total_used_mb + expert_mb > total_budget_mb:
            break
        device = layer_device_fn(layer)
        dev_idx = device.index if device.index is not None else 0
        if dev_idx in oom_devices:
            continue
        if used_mb[dev_idx] + expert_mb > dev_budget_mb[dev_idx]:
            continue
        try:
            raw_np = packed_loader.load_expert_raw_bytes(layer, eid)
            gpu_tensor = torch.from_numpy(raw_np).to(device, non_blocking=True)
            cache[(layer, eid)] = gpu_tensor
            used_mb[dev_idx] += expert_mb
            total_used_mb += expert_mb
        except torch.cuda.OutOfMemoryError:
            oom_devices.add(dev_idx)
            torch.cuda.empty_cache()
            continue

    per_dev = {i: f"GPU {i}: {used_mb[i]/1024:.2f}GB" for i in range(n_dev)}
    covered = sum(cnt for k, cnt in activation_trace.items() if k in cache)
    total_obs = sum(activation_trace.values())
    print(f"GPU cache: {len(cache)} experts, {total_used_mb/1024:.1f}GB total")
    for i in range(n_dev):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        print(f"  GPU {i}: {alloc:.2f}GB allocated (+{used_mb[i]/1024:.2f}GB cache)")
    print(f"GPU cache covers {covered}/{total_obs} = {covered/total_obs:.1%}")
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
    parser.add_argument('--2bit', dest='two_bit', action='store_true',
                        help='Use 2-bit experts (3.38MB/exp, 7x smaller than bf16)')
    parser.add_argument('--float8-nonexpert', action='store_true',
                        help='Store non-expert Linear weights as float8 (halves attention VRAM)')
    parser.add_argument('--gpu-hot-gb', type=float, default=0.0,
                        help='GB of GPU VRAM to use for resident 2-bit expert cache (0=disabled)')
    parser.add_argument('--gpu-margin-gb', type=float, default=2.5,
                        help='GB of VRAM to reserve as safety margin when building GPU cache')
    parser.add_argument('--cache-format', default='auto',
                        choices=['auto', 'bf16', 'raw'],
                        help='HOT cache format (bf16=dequanted 24MB/exp, raw=GPTQ 6.5MB/exp)')
    parser.add_argument('--compile', action='store_true',
                        help='Apply torch.compile to non-MoE layers (reduce kernel launch overhead)')
    parser.add_argument('--prefetch', action='store_true',
                        help='Enable cross-layer expert prefetch (H2D during attention window)')
    parser.add_argument('--save-hs', action='store_true',
                        help='Save hidden states during COLD pass for predictor training')
    parser.add_argument('--hs-out', default='/tmp/hs_trace_397b.pkl',
                        help='Output path for hidden state trace (--save-hs)')
    parser.add_argument('--collect-prompts', default=None,
                        help='File with extra prompts (one per line) to collect hs from')
    parser.add_argument('--collect-tokens', type=int, default=None,
                        help='Tokens per prompt during hs collection (default: same as --tokens)')
    parser.add_argument('--predictor', default=None,
                        help='Path to trained CrossLayerExpertPredictor .pt file')
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

    # Split: fewer layers on GPU 0 (already heavy with embed/lm_head) → more on GPU 1
    # 24/36 leaves ~1GB free on GPU 0 for expert streaming
    def layer_device(idx):
        return torch.device('cuda:0') if idx < 24 else torch.device('cuda:1')

    print("3. Loading non-expert weights...")
    # Non-expert safetensors live in GPTQ dir (original MODEL_DIR has no safetensors)
    nonexpert_src = GPTQ_DIR if not os.path.exists(
        os.path.join(MODEL_DIR, 'model.safetensors.index.json')) else MODEL_DIR
    shards = ShardManager(nonexpert_src)
    load_non_expert_weights(model, shards, layer_device, verbose=True)
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.memory_allocated(i)/1e9:.2f}GB")

    if args.float8_nonexpert:
        from float8_nonexpert import apply_float8_nonexpert
        print("3b. Applying float8 non-expert quantization...")
        apply_float8_nonexpert(model)
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i} after float8: {torch.cuda.memory_allocated(i)/1e9:.2f}GB")

    print("4. Attaching layer device hooks...")
    attach_layer_device_hooks(model, layer_device)

    is_2bit = args.two_bit
    if is_2bit:
        print("5. Setting up TwoBitLoader (2-bit, 3.38MB/expert)...")
        packed_loader = TwoBitLoader(PACKED_2BIT_DIR, device=torch.device('cuda:0'))
    elif args.gptq:
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
    n_patched = patch_experts(model, packed_loader, hot_cache, gpu_cache)
    print(f"  Patched {n_patched} MoE layers")

    if args.compile:
        print("6b. Applying torch.compile (reduce-overhead, dynamic)...")
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True
        model = torch.compile(model, mode='reduce-overhead', dynamic=True)
        print("  torch.compile applied (first run will be slow for tracing)")

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

    def run_once(label):
        if s:
            for k in s: s[k] = 0 if isinstance(s[k], int) else 0.0
        t = time.time()
        with torch.no_grad():
            out = model.generate(input_ids, attention_mask=attention_mask,
                                 max_new_tokens=args.tokens, do_sample=False,
                                 use_cache=True)
        el = time.time() - t
        nt = out.shape[1] - input_ids.shape[1]
        resp = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n=== [{label}] {nt} tokens in {el:.1f}s ({nt/el:.3f} tok/s) ===")
        print(f"Output: {resp}")
        if s:
            total_ops = s['gpu_hits'] + s['hits'] + s['misses']
            print(f"Breakdown: gpu_hit={s['gpu_hits']}×, cpu_hit={s['hits']}×, "
                  f"miss={s['misses']}× ({s['load_miss_s']:.1f}s), compute={s['compute_s']:.1f}s")
            if total_ops:
                print(f"  gpu_hit_rate={s['gpu_hits']/total_ops:.1%}  "
                      f"cpu_hit_rate={s['hits']/total_ops:.1%}  "
                      f"miss_rate={s['misses']/total_ops:.1%}")
            if s['misses']:
                print(f"  avg miss: {s['load_miss_s']/s['misses']*1000:.1f}ms/expert")

    # Pass 1: COLD, traces activations
    if args.save_hs:
        packed_loader._collect_hs[0] = True
    run_once("COLD (trace)")
    trace = getattr(packed_loader, '_trace', {})
    print(f"Traced {len(trace)} unique (layer, expert) activations")

    packed_loader._collect_hs[0] = False  # stop collecting after COLD pass

    # Collect hidden states from additional diverse prompts
    if args.save_hs and args.collect_prompts:
        col_tokens = args.collect_tokens or args.tokens
        extra_prompts = []
        with open(args.collect_prompts) as f:
            for line in f:
                line = line.strip()
                if line and line != args.prompt:
                    extra_prompts.append(line)
        print(f"Collecting hs from {len(extra_prompts)} extra prompts "
              f"({col_tokens} tokens each)...")
        packed_loader._collect_hs[0] = True
        for i, ep in enumerate(extra_prompts):
            msgs = [{"role": "user", "content": ep}]
            etxt = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True,
                                                  enable_thinking=False)
            etok = tokenizer(etxt, return_tensors="pt")
            eids = etok.input_ids.to('cuda:0')
            emask = etok.attention_mask.to('cuda:0')
            with torch.no_grad():
                model.generate(eids, attention_mask=emask,
                                max_new_tokens=col_tokens, do_sample=False,
                                use_cache=True)
            print(f"  [{i+1}/{len(extra_prompts)}] collected, "
                  f"total samples: {len(packed_loader._hs_trace)}")
        packed_loader._collect_hs[0] = False

    # Save hidden state trace for offline predictor training
    if args.save_hs and packed_loader._hs_trace:
        import pickle
        with open(args.hs_out, 'wb') as f:
            pickle.dump(packed_loader._hs_trace, f)
        print(f"Saved {len(packed_loader._hs_trace)} hidden state samples → {args.hs_out}")

    # Load trained predictor and enable prefetch
    if args.predictor:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from train_cross_predictor import CrossLayerExpertPredictor
        pred = CrossLayerExpertPredictor(hidden_dim=4096, num_experts=512, num_layers=60)
        pred.load_state_dict(torch.load(args.predictor, map_location='cpu'))
        pred.eval()
        packed_loader._predictor[0] = pred
        packed_loader._prefetch_enabled[0] = True
        print(f"Loaded predictor from {args.predictor} — prefetch ENABLED")

    # Flush last token's activations (flushed at layer_idx==0 of *next* token)
    if packed_loader._cur_acts[0]:
        packed_loader._token_acts.append(dict(packed_loader._cur_acts[0]))
        packed_loader._cur_acts[0].clear()

    # Build cross-layer correlation table from COLD activations
    token_acts = packed_loader._token_acts
    if token_acts:
        from collections import Counter, defaultdict
        _layer_table = defaultdict(Counter)
        for tok in token_acts:
            for layer in range(tc.num_hidden_layers - 1):
                if layer in tok and (layer + 1) in tok:
                    for eid_L in tok[layer]:
                        for eid_L1 in tok[layer + 1]:
                            _layer_table[(layer, eid_L)][eid_L1] += 1
        cross_table = {k: [e for e, _ in v.most_common(4)]
                       for k, v in _layer_table.items()}
        packed_loader._cross_table[0] = cross_table

        # Measure predicted hit rate against observed activations
        total_pred = total_hit = 0
        for tok in token_acts:
            for layer in range(tc.num_hidden_layers - 1):
                if layer in tok and (layer + 1) in tok:
                    actual_L1 = tok[layer + 1]
                    for eid_L in tok[layer]:
                        preds = set(cross_table.get((layer, eid_L), []))
                        total_pred += len(preds)
                        total_hit += len(preds & actual_L1)
        hit_rate = total_hit / total_pred if total_pred else 0.0
        print(f"Cross-layer prediction: {total_hit}/{total_pred} = {hit_rate:.1%} hit rate "
              f"({len(token_acts)} tokens, top-4 per active expert)")
        if args.prefetch:
            packed_loader._prefetch_enabled[0] = True
            print("Cross-layer prefetch: ENABLED")

    # Build GPU resident cache (2-bit raw bytes on VRAM) if requested
    if args.gpu_hot_gb > 0 and is_2bit:
        torch.cuda.empty_cache()
        built_gpu = build_gpu_hot_cache_2bit(trace, packed_loader, layer_device,
                                              max_vram_gb=args.gpu_hot_gb,
                                              safety_margin_gb=args.gpu_margin_gb)
        gpu_cache.update(built_gpu)
        run_once("GPU_ONLY")

    # Build CPU pinned HOT cache from trace if requested
    if args.hot_pct > 0:
        total_slots = int(tc.num_hidden_layers * tc.num_experts * args.hot_pct)
        built, cache_fmt = build_hot_cache_from_trace(trace, packed_loader, total_slots,
                                                       cache_format=args.cache_format)
        hot_cache.update(built)
        hot_cache['__format__'] = cache_fmt
        # Compute coverage: what fraction of observed activations are in cache?
        covered = sum(cnt for k, cnt in trace.items() if k in hot_cache)
        total_obs = sum(trace.values())
        print(f"HOT cache covers {covered}/{total_obs} = {covered/total_obs:.1%} of observed activations")

    label = "FULL+GPU" if (gpu_cache and hot_cache) else ("GPU" if gpu_cache else "HOT")
    if args.prefetch:
        label += "+PF"
    run_once(label)
    if args.prefetch:
        ps = packed_loader._prefetch_stats
        if ps[1]:
            print(f"Prefetch: {ps[0]} used / {ps[1]} started = {ps[0]/ps[1]:.1%} use rate")

    # Cleanup
    shards.close()
    if hasattr(packed_loader, 'close'):
        packed_loader.close()


if __name__ == '__main__':
    main()
