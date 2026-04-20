"""Memory-mapped expert loader using safetensors mmap support.

Eliminates safetensors parsing overhead by memory-mapping the file
and computing tensor offsets directly from the header.
"""
import json
import os
import struct
import mmap
import numpy as np
import time


class MmapExpertLoader:
    """Load expert weights via mmap for zero-copy SSD access.

    Uses safetensors file format directly — reads header once,
    then accesses expert data via mmap offset without parsing.
    """

    def __init__(self, model_dir, dtype='bf16'):
        """
        Args:
            model_dir: path to model directory with safetensors files
            dtype: 'bf16' (original) or 'f32' (converted)
        """
        self.model_dir = model_dir
        self.dtype = dtype

        with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
            self.index = json.load(f)

        # Parse safetensors headers and build offset map
        self._mmaps = {}  # filename -> mmap object
        self._fd_map = {}  # filename -> fd
        self._header_cache = {}  # filename -> parsed header
        self._data_offset = {}  # filename -> data start offset

    def _ensure_mmap(self, filename):
        """Open and mmap a safetensors file if not already."""
        if filename in self._mmaps:
            return

        path = os.path.join(self.model_dir, filename)
        fd = os.open(path, os.O_RDONLY)
        file_size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)

        # Parse safetensors header
        header_size = struct.unpack('<Q', mm[:8])[0]
        header_json = mm[8:8 + header_size].decode('utf-8')
        header = json.loads(header_json)

        self._fd_map[filename] = fd
        self._mmaps[filename] = mm
        self._header_cache[filename] = header
        self._data_offset[filename] = 8 + header_size

    def _get_tensor_raw(self, key):
        """Get raw bytes + metadata for a tensor key via mmap."""
        filename = self.index['weight_map'][key]
        self._ensure_mmap(filename)

        header = self._header_cache[filename]
        meta = header[key]
        data_start = self._data_offset[filename]

        begin, end = meta['data_offsets']
        raw = self._mmaps[filename][data_start + begin:data_start + end]

        return raw, meta['shape'], meta['dtype']

    def load_expert_gate_up(self, layer, expert_id):
        """Load gate_up_proj for one expert as float32 numpy array."""
        key = f'model.language_model.layers.{layer}.mlp.experts.gate_up_proj'
        raw, shape, dtype = self._get_tensor_raw(key)

        # shape = [num_experts, out_dim, in_dim]
        num_experts, out_dim, in_dim = shape
        bytes_per_element = 2 if dtype == 'BF16' else 4
        expert_size = out_dim * in_dim * bytes_per_element
        offset = expert_id * expert_size

        expert_raw = raw[offset:offset + expert_size]

        if dtype == 'BF16':
            return self._bf16_to_f32(expert_raw, (out_dim, in_dim))
        else:
            return np.frombuffer(expert_raw, dtype=np.float32).reshape(out_dim, in_dim).copy()

    def load_expert_down(self, layer, expert_id):
        """Load down_proj for one expert as float32 numpy array."""
        key = f'model.language_model.layers.{layer}.mlp.experts.down_proj'
        raw, shape, dtype = self._get_tensor_raw(key)

        num_experts, out_dim, in_dim = shape
        bytes_per_element = 2 if dtype == 'BF16' else 4
        expert_size = out_dim * in_dim * bytes_per_element
        offset = expert_id * expert_size

        expert_raw = raw[offset:offset + expert_size]

        if dtype == 'BF16':
            return self._bf16_to_f32(expert_raw, (out_dim, in_dim))
        else:
            return np.frombuffer(expert_raw, dtype=np.float32).reshape(out_dim, in_dim).copy()

    @staticmethod
    def _bf16_to_f32(raw_bytes, shape):
        """Convert raw bf16 bytes to float32 numpy array."""
        u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
        f32_bits = u16.astype(np.uint32) << 16
        return np.frombuffer(f32_bits.tobytes(), dtype=np.float32).reshape(shape).copy()

    def benchmark_load(self, layer=0, n_experts=20):
        """Benchmark expert loading speed."""
        # Warmup
        self.load_expert_gate_up(layer, 0)
        self.load_expert_down(layer, 0)

        ids = list(range(n_experts))

        t0 = time.perf_counter()
        for eid in ids:
            self.load_expert_gate_up(layer, eid)
        gate_up_ms = (time.perf_counter() - t0) / n_experts * 1000

        t0 = time.perf_counter()
        for eid in ids:
            self.load_expert_down(layer, eid)
        down_ms = (time.perf_counter() - t0) / n_experts * 1000

        return {
            'gate_up_ms': gate_up_ms,
            'down_ms': down_ms,
            'total_ms': gate_up_ms + down_ms,
        }

    def close(self):
        for mm in self._mmaps.values():
            mm.close()
        for fd in self._fd_map.values():
            os.close(fd)
        self._mmaps.clear()
        self._fd_map.clear()
