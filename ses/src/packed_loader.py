"""Fast packed binary expert loader with mmap + direct bf16 GPU transfer.

Loads experts from packed_experts_bf16/layer_XX.bin files
via mmap → bf16 tensor → CUDA transfer.

~1ms per expert (vs 13ms with safetensors).
"""
import json
import mmap
import os
import time

import numpy as np
import torch


class PackedExpertLoader:
    """Load experts from packed binary files via mmap.

    Each layer file: 256 experts × EXPERT_SIZE bytes contiguous.
    Expert layout: [gate_up_proj bf16][down_proj bf16]
    """

    def __init__(self, packed_dir, device=None):
        layout_path = os.path.join(packed_dir, 'layout.json')
        with open(layout_path) as f:
            self.layout = json.load(f)

        self.expert_size = self.layout['expert_size']
        self.gu_bytes = self.layout['gate_up_bytes']
        self.dw_bytes = self.layout['down_bytes']
        self.gu_shape = tuple(self.layout['gate_up_shape'])
        self.dw_shape = tuple(self.layout['down_shape'])
        self.device = device or torch.device('cpu')

        self._mmaps = {}  # layer_idx -> (mmap, fd)
        self._packed_dir = packed_dir

    def _ensure_mmap(self, layer):
        if layer not in self._mmaps:
            path = os.path.join(self._packed_dir, f'layer_{layer:02d}.bin')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Packed file not found: {path}")
            fd = os.open(path, os.O_RDONLY)
            file_size = os.fstat(fd).st_size
            mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
            self._mmaps[layer] = (mm, fd)
        return self._mmaps[layer][0]

    def load_expert(self, layer, expert_id, device=None):
        """Load one expert as bf16 tensors on target device.

        Returns:
            gate_up: [1024, 2048] bf16 tensor
            down: [2048, 512] bf16 tensor
        """
        dev = device or self.device
        mm = self._ensure_mmap(layer)
        offset = expert_id * self.expert_size

        # Read raw bf16 bytes
        gu_raw = mm[offset:offset + self.gu_bytes]
        dw_raw = mm[offset + self.gu_bytes:offset + self.expert_size]

        # numpy uint16 → torch bf16 → device
        gu = torch.from_numpy(
            np.frombuffer(gu_raw, dtype=np.uint16).reshape(self.gu_shape).copy()
        ).view(torch.bfloat16).to(dev)

        dw = torch.from_numpy(
            np.frombuffer(dw_raw, dtype=np.uint16).reshape(self.dw_shape).copy()
        ).view(torch.bfloat16).to(dev)

        return gu, dw

    def load_experts_batch(self, layer, expert_ids, device=None):
        """Load multiple experts for one layer."""
        return [self.load_expert(layer, eid, device) for eid in expert_ids]

    def preload_hot_experts(self, hot_ids_per_layer, device=None):
        """Pre-load HOT experts into GPU/CPU memory.

        Args:
            hot_ids_per_layer: {layer: [expert_ids]}

        Returns:
            cache: {(layer, expert_id): (gate_up_tensor, down_tensor)}
        """
        dev = device or self.device
        cache = {}
        total_bytes = 0
        t0 = time.time()

        for layer, expert_ids in hot_ids_per_layer.items():
            for eid in expert_ids:
                gu, dw = self.load_expert(layer, eid, dev)
                cache[(layer, eid)] = (gu, dw)
                total_bytes += gu.nelement() * 2 + dw.nelement() * 2  # bf16

        elapsed = time.time() - t0
        print(f"HOT cache: {len(cache)} experts, {total_bytes/1e9:.1f}GB, "
              f"{elapsed:.1f}s ({total_bytes/elapsed/1e9:.1f} GB/s)")
        return cache

    def benchmark(self, layer=0, n_experts=50, device=None):
        """Benchmark expert loading speed."""
        dev = device or self.device
        # Warmup
        self.load_expert(layer, 0, dev)

        times = []
        num_experts = self.layout.get('num_experts', 256)
        for i in range(n_experts):
            torch.cuda.synchronize() if dev.type == 'cuda' else None
            t0 = time.perf_counter()
            self.load_expert(layer, i % num_experts, dev)
            torch.cuda.synchronize() if dev.type == 'cuda' else None
            times.append((time.perf_counter() - t0) * 1000)

        return {
            'mean_ms': float(np.mean(times)),
            'p50_ms': float(np.percentile(times, 50)),
            'p99_ms': float(np.percentile(times, 99)),
            '8_experts_ms': float(np.mean(times) * 8),
        }

    def close(self):
        for mm, fd in self._mmaps.values():
            mm.close()
            os.close(fd)
        self._mmaps.clear()
