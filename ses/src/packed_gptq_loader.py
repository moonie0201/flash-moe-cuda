"""Fast packed GPTQ expert loader.

Reads from packed_experts/layer_XX.bin files where each expert is 6.27MB
of contiguous bytes. Single pread + single CPU→GPU transfer per expert.

Layout per expert (byte offsets):
  gate_proj: qweight(2MB) + qzeros(16KB) + scales(64KB) + g_idx(16KB)
  up_proj:   qweight(2MB) + qzeros(16KB) + scales(64KB) + g_idx(16KB)
  down_proj: qweight(2MB) + qzeros(16KB) + scales(64KB) + g_idx(4KB)
Total: 6,574,080 bytes per expert.
"""
import json
import mmap
import os

import numpy as np
import torch

from gptq_expert_loader import gptq_dequant


class PackedGPTQLoader:
    """Load packed GPTQ experts via mmap, one contiguous byte-block per expert."""

    # Component offsets (matches repack_397b_gptq.py layout)
    GATE_UP_QW_BYTES = 2_097_152
    GATE_UP_QZ_BYTES = 16_384
    GATE_UP_SC_BYTES = 65_536
    GATE_UP_GI_BYTES = 16_384
    GATE_UP_PROJ_BYTES = GATE_UP_QW_BYTES + GATE_UP_QZ_BYTES + GATE_UP_SC_BYTES + GATE_UP_GI_BYTES

    DOWN_QW_BYTES = 2_097_152
    DOWN_QZ_BYTES = 16_384
    DOWN_SC_BYTES = 65_536
    DOWN_GI_BYTES = 4_096
    DOWN_PROJ_BYTES = DOWN_QW_BYTES + DOWN_QZ_BYTES + DOWN_SC_BYTES + DOWN_GI_BYTES

    EXPERT_SIZE = 2 * GATE_UP_PROJ_BYTES + DOWN_PROJ_BYTES  # 6,574,080

    # Shapes
    GATE_UP_QW_SHAPE = (512, 1024)
    GATE_UP_QZ_SHAPE = (32, 128)
    GATE_UP_SC_SHAPE = (32, 1024)
    GATE_UP_GI_SHAPE = (4096,)
    DOWN_QW_SHAPE = (128, 4096)
    DOWN_QZ_SHAPE = (8, 512)
    DOWN_SC_SHAPE = (8, 4096)
    DOWN_GI_SHAPE = (1024,)

    def __init__(self, packed_dir, device=None):
        self.packed_dir = packed_dir
        self.device = device or torch.device('cuda:0')
        self._mmaps = {}
        with open(os.path.join(packed_dir, 'layout.json')) as f:
            self.layout = json.load(f)

    def _ensure_mmap(self, layer):
        if layer not in self._mmaps:
            path = os.path.join(self.packed_dir, f'layer_{layer:02d}.bin')
            fd = os.open(path, os.O_RDONLY)
            size = os.fstat(fd).st_size
            mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
            self._mmaps[layer] = (mm, fd)
        return self._mmaps[layer][0]

    def load_expert_raw(self, layer, expert_id):
        """Read one expert's 6.27MB contiguous block. Returns single bytes-like.

        Returns a single bytes buffer that can be transferred to GPU as one chunk.
        """
        mm = self._ensure_mmap(layer)
        off = expert_id * self.EXPERT_SIZE
        return mm[off:off + self.EXPERT_SIZE]

    def _split_raw(self, raw_bytes):
        """Slice a raw expert bytes block into the 12 components (still on CPU).

        Returns a tuple of 3 per-projection tuples (gate, up, down):
          (qweight, qzeros, scales, g_idx)
        """
        GP = self.GATE_UP_PROJ_BYTES
        DQW = self.DOWN_QW_BYTES
        DQZ = self.DOWN_QZ_BYTES
        DSC = self.DOWN_SC_BYTES
        DGI = self.DOWN_GI_BYTES
        GQW = self.GATE_UP_QW_BYTES
        GQZ = self.GATE_UP_QZ_BYTES
        GSC = self.GATE_UP_SC_BYTES
        GGI = self.GATE_UP_GI_BYTES

        def proj_slices(base, qw_size, qz_size, sc_size, gi_size):
            o = base
            qw = raw_bytes[o:o+qw_size]; o += qw_size
            qz = raw_bytes[o:o+qz_size]; o += qz_size
            sc = raw_bytes[o:o+sc_size]; o += sc_size
            gi = raw_bytes[o:o+gi_size]
            return qw, qz, sc, gi

        gate = proj_slices(0, GQW, GQZ, GSC, GGI)
        up   = proj_slices(GP, GQW, GQZ, GSC, GGI)
        down = proj_slices(2 * GP, DQW, DQZ, DSC, DGI)
        return gate, up, down

    @staticmethod
    def _bytes_to_tensor(buf, dtype, shape, device):
        if dtype == 'i32':
            np_arr = np.frombuffer(buf, dtype=np.int32)
        elif dtype == 'f16':
            np_arr = np.frombuffer(buf, dtype=np.uint16)
        else:
            raise ValueError(dtype)
        np_arr = np_arr.reshape(shape).copy()
        t = torch.from_numpy(np_arr)
        if dtype == 'f16':
            t = t.view(torch.float16)
        return t.to(device)

    def _raw_to_tensors_gpu(self, raw_bytes, device):
        """Load all 12 tensors to GPU from raw bytes in single transfer."""
        # raw_bytes can be: mmap bytes, numpy uint8 array, or torch uint8 tensor
        if isinstance(raw_bytes, torch.Tensor):
            buf_cpu = raw_bytes
        elif isinstance(raw_bytes, np.ndarray):
            buf_cpu = torch.from_numpy(raw_bytes)
        else:
            buf_np = np.frombuffer(raw_bytes, dtype=np.uint8).copy()
            buf_cpu = torch.from_numpy(buf_np)
        # Transfer whole buffer to GPU in one shot
        buf_gpu = buf_cpu.to(device, non_blocking=True)

        def slice_and_view(start, num_elements, target_dtype, shape):
            raw_slice = buf_gpu[start:start + num_elements * target_dtype.itemsize if target_dtype != torch.float16 else start + num_elements * 2]
            return raw_slice.view(target_dtype).reshape(shape)

        GP = self.GATE_UP_PROJ_BYTES
        # gate_proj
        off = 0
        gate_qw = buf_gpu[off:off + self.GATE_UP_QW_BYTES].view(torch.int32).reshape(self.GATE_UP_QW_SHAPE)
        off += self.GATE_UP_QW_BYTES
        gate_qz = buf_gpu[off:off + self.GATE_UP_QZ_BYTES].view(torch.int32).reshape(self.GATE_UP_QZ_SHAPE)
        off += self.GATE_UP_QZ_BYTES
        gate_sc = buf_gpu[off:off + self.GATE_UP_SC_BYTES].view(torch.float16).reshape(self.GATE_UP_SC_SHAPE)
        off += self.GATE_UP_SC_BYTES
        gate_gi = buf_gpu[off:off + self.GATE_UP_GI_BYTES].view(torch.int32).reshape(self.GATE_UP_GI_SHAPE)

        # up_proj
        off = GP
        up_qw = buf_gpu[off:off + self.GATE_UP_QW_BYTES].view(torch.int32).reshape(self.GATE_UP_QW_SHAPE)
        off += self.GATE_UP_QW_BYTES
        up_qz = buf_gpu[off:off + self.GATE_UP_QZ_BYTES].view(torch.int32).reshape(self.GATE_UP_QZ_SHAPE)
        off += self.GATE_UP_QZ_BYTES
        up_sc = buf_gpu[off:off + self.GATE_UP_SC_BYTES].view(torch.float16).reshape(self.GATE_UP_SC_SHAPE)
        off += self.GATE_UP_SC_BYTES
        up_gi = buf_gpu[off:off + self.GATE_UP_GI_BYTES].view(torch.int32).reshape(self.GATE_UP_GI_SHAPE)

        # down_proj
        off = 2 * GP
        down_qw = buf_gpu[off:off + self.DOWN_QW_BYTES].view(torch.int32).reshape(self.DOWN_QW_SHAPE)
        off += self.DOWN_QW_BYTES
        down_qz = buf_gpu[off:off + self.DOWN_QZ_BYTES].view(torch.int32).reshape(self.DOWN_QZ_SHAPE)
        off += self.DOWN_QZ_BYTES
        down_sc = buf_gpu[off:off + self.DOWN_SC_BYTES].view(torch.float16).reshape(self.DOWN_SC_SHAPE)
        off += self.DOWN_SC_BYTES
        down_gi = buf_gpu[off:off + self.DOWN_GI_BYTES].view(torch.int32).reshape(self.DOWN_GI_SHAPE)

        return (
            (gate_qw, gate_qz, gate_sc, gate_gi),
            (up_qw,   up_qz,   up_sc,   up_gi),
            (down_qw, down_qz, down_sc, down_gi),
        )

    def load_expert(self, layer, expert_id, device=None):
        """Load and dequant one expert. Returns (gate_up_w [2*inter, hidden], down_w [hidden, inter])."""
        dev = device or self.device
        raw = self.load_expert_raw(layer, expert_id)

        if dev.type == 'cuda':
            gate_r, up_r, down_r = self._raw_to_tensors_gpu(raw, dev)
        else:
            raise NotImplementedError("CPU path not implemented for PackedGPTQLoader")

        gate = gptq_dequant(*gate_r)
        up   = gptq_dequant(*up_r)
        down = gptq_dequant(*down_r)

        gate_up_weight = torch.cat([gate, up], dim=1).transpose(0, 1).contiguous()
        down_weight = down.transpose(0, 1).contiguous()
        return gate_up_weight, down_weight

    def load_expert_raw_bytes(self, layer, expert_id):
        """Return raw bytes as a single numpy uint8 array (for CPU caching)."""
        raw = self.load_expert_raw(layer, expert_id)
        return np.frombuffer(raw, dtype=np.uint8).copy()

    def dequant_raw_bytes_to_gpu(self, raw_bytes, device):
        """Dequant from cached raw bytes → F.linear weights on GPU."""
        gate_r, up_r, down_r = self._raw_to_tensors_gpu(raw_bytes, device)
        gate = gptq_dequant(*gate_r)
        up   = gptq_dequant(*up_r)
        down = gptq_dequant(*down_r)
        gate_up_weight = torch.cat([gate, up], dim=1).transpose(0, 1).contiguous()
        down_weight = down.transpose(0, 1).contiguous()
        return gate_up_weight, down_weight

    def close(self):
        for mm, fd in self._mmaps.values():
            mm.close()
            os.close(fd)
        self._mmaps.clear()
