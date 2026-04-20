"""GPTQ-Int4 expert loader — reads qweight/qzeros/scales/g_idx from safetensors.

Dequantizes to bf16 on GPU, returns weight tensors ready for F.linear.

GPTQ format (per projection):
  qweight: I32 [in/8, out]   — 4-bit packed along input axis
  qzeros:  I32 [groups, out/8] — 4-bit packed zero points, per group
  scales:  F16 [groups, out]   — scale per group per output channel
  g_idx:   I32 [in]            — group index for each input feature

Dequant: w[i, j] = (qweight_unpack[i, j] - qzeros_unpack[g_idx[i], j]) * scales[g_idx[i], j]
"""
import json
import mmap
import os
import struct

import numpy as np
import torch


def _unpack_along_rows(packed):
    """I32 [N, M] → int [N*8, M]: unpack 8 nibbles per row along axis 0."""
    N, M = packed.shape
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)  # [8]
    unpacked = (packed.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF  # [N, 8, M]
    return unpacked.reshape(N * 8, M)


def _unpack_along_cols(packed):
    """I32 [N, M] → int [N, M*8]: unpack 8 nibbles per element along axis 1."""
    N, M = packed.shape
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)  # [8]
    unpacked = (packed.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF  # [N, M, 8]
    return unpacked.reshape(N, M * 8)


def gptq_dequant(qweight, qzeros, scales, g_idx, dtype=torch.bfloat16):
    """Dequantize GPTQ tensors to dense weight matrix [in, out].

    Args:
        qweight: I32 [in/8, out]
        qzeros:  I32 [groups, out/8]
        scales:  F16 [groups, out]
        g_idx:   I32 [in]

    Returns:
        W: [in, out] bf16
    """
    qw = _unpack_along_rows(qweight).to(torch.int32)   # [in, out]
    qz = _unpack_along_cols(qzeros).to(torch.int32)    # [groups, out]

    # Standard GPTQ uses sym=True with qzeros offset; GPTQ v2 adds +1 to zero.
    # Qwen GPTQ config says sym=True, so zero is typically 2^(bits-1) - 1 = 7
    # But actual implementation: dequant = (q - (qzero + 1)) * scale

    # For each input i, group = g_idx[i]
    g = g_idx.to(torch.long)  # [in]
    zeros_per_input = qz[g]     # [in, out]
    scales_per_input = scales[g].to(torch.float32)  # [in, out]

    # sym=True: stored zero is already the exact midpoint (=8), no +1 offset
    W = (qw - zeros_per_input).to(torch.float32) * scales_per_input  # [in, out]
    return W.to(dtype)


class GPTQExpertLoader:
    """Loads GPTQ expert tensors from safetensors via mmap.

    Each expert has 3 projections (gate_proj, up_proj, down_proj) × 4 tensors.
    """

    def __init__(self, model_dir, device=None):
        self.model_dir = model_dir
        self.device = device or torch.device('cuda:0')
        with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
            self.index = json.load(f)['weight_map']
        self._shards = {}

    def _mmap_shard(self, shard):
        if shard not in self._shards:
            path = os.path.join(self.model_dir, shard)
            fd = os.open(path, os.O_RDONLY)
            size = os.fstat(fd).st_size
            mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
            hs = struct.unpack('<Q', mm[:8])[0]
            hdr = json.loads(mm[8:8+hs].decode('utf-8'))
            self._shards[shard] = (mm, fd, 8+hs, hdr)
        return self._shards[shard]

    def _load_tensor(self, key, device):
        shard = self.index[key]
        mm, fd, do, hdr = self._mmap_shard(shard)
        meta = hdr[key]
        start, end = meta['data_offsets']
        raw = mm[do + start:do + end]
        shape = meta['shape']
        dt = meta['dtype']
        if dt == 'I32':
            arr = np.frombuffer(raw, dtype=np.int32).reshape(shape).copy()
            return torch.from_numpy(arr).to(device)
        elif dt == 'F16':
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape).copy()
            t = torch.from_numpy(arr).view(torch.float16)
            return t.to(device)
        elif dt == 'BF16':
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape).copy()
            t = torch.from_numpy(arr).view(torch.bfloat16)
            return t.to(device)
        else:
            raise ValueError(f'Unsupported dtype: {dt}')

    def load_proj_raw(self, layer, expert_id, proj_name, device='cpu'):
        """Load raw GPTQ tensors for one projection (no dequant).

        Returns (qweight, qzeros, scales, g_idx) on the given device.
        Raw GPTQ tensors are small (~2MB total per projection).
        """
        dev = torch.device(device) if isinstance(device, str) else device
        prefix = f'model.language_model.layers.{layer}.mlp.experts.{expert_id}.{proj_name}'
        return (
            self._load_tensor(f'{prefix}.qweight', dev),
            self._load_tensor(f'{prefix}.qzeros', dev),
            self._load_tensor(f'{prefix}.scales', dev),
            self._load_tensor(f'{prefix}.g_idx', dev),
        )

    def load_proj(self, layer, expert_id, proj_name, device=None):
        """Load and dequant one projection → W [in, out] on device."""
        dev = device or self.device
        qw, qz, sc, gi = self.load_proj_raw(layer, expert_id, proj_name, dev)
        return gptq_dequant(qw, qz, sc, gi)

    def _dequant_and_pack(self, gate_raw, up_raw, down_raw, gpu_dev):
        """Dequant raw tensors on GPU, return F.linear-ready weights."""
        # Move raw to GPU if not already
        def tog(tup):
            return tuple(t.to(gpu_dev) if t.device != gpu_dev else t for t in tup)
        g_qw, g_qz, g_sc, g_gi = tog(gate_raw)
        u_qw, u_qz, u_sc, u_gi = tog(up_raw)
        d_qw, d_qz, d_sc, d_gi = tog(down_raw)

        gate = gptq_dequant(g_qw, g_qz, g_sc, g_gi)
        up   = gptq_dequant(u_qw, u_qz, u_sc, u_gi)
        down = gptq_dequant(d_qw, d_qz, d_sc, d_gi)

        # [hidden, inter] cat along inter → [hidden, 2*inter], transpose → [2*inter, hidden]
        gate_up_weight = torch.cat([gate, up], dim=1).transpose(0, 1).contiguous()
        down_weight = down.transpose(0, 1).contiguous()
        return gate_up_weight, down_weight

    def load_expert(self, layer, expert_id, device=None):
        """Load expert → F.linear-ready (gate_up_w [2*inter, hidden], down_w [hidden, inter]).

        If device is GPU: load raw to CPU (small), then dequant on GPU.
        If device is CPU: load raw + dequant on CPU (slow — avoid).
        """
        dev = device or self.device
        gate_raw = self.load_proj_raw(layer, expert_id, 'gate_proj', 'cpu')
        up_raw   = self.load_proj_raw(layer, expert_id, 'up_proj',   'cpu')
        down_raw = self.load_proj_raw(layer, expert_id, 'down_proj', 'cpu')

        if dev.type == 'cuda':
            return self._dequant_and_pack(gate_raw, up_raw, down_raw, dev)
        # CPU fallback (slow)
        return self._dequant_and_pack(gate_raw, up_raw, down_raw, torch.device('cpu'))

    def load_expert_raw(self, layer, expert_id):
        """Load raw GPTQ tensors to CPU for caching. Returns tuple of 3 raw-proj tuples."""
        return (
            self.load_proj_raw(layer, expert_id, 'gate_proj', 'cpu'),
            self.load_proj_raw(layer, expert_id, 'up_proj',   'cpu'),
            self.load_proj_raw(layer, expert_id, 'down_proj', 'cpu'),
        )

    def dequant_raw_to_gpu(self, raw_tuple, gpu_dev):
        """Dequant previously-cached raw tensors on GPU."""
        return self._dequant_and_pack(raw_tuple[0], raw_tuple[1], raw_tuple[2], gpu_dev)

    def close(self):
        for mm, fd, _, _ in self._shards.values():
            mm.close()
            os.close(fd)
