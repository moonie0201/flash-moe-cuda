"""2-bit packed expert loader with GPU dequantization.

Reads from packed_experts_2bit/layer_XX.bin (3.38MB/expert).
Dequantizes on GPU using vectorized torch ops — no custom CUDA needed.

Format per expert:
  gate_up_codes  [2,097,152 bytes]  uint8, 4 codes/byte, shape encodes [2048, 4096]
  gate_up_scales [  262,144 bytes]  fp16, shape [2048, 64]
  down_codes     [1,048,576 bytes]  uint8, 4 codes/byte, shape encodes [4096, 1024]
  down_scales    [  131,072 bytes]  fp16, shape [4096, 16]

Dequant: w = (code - 1.5) * scale  where code in {0,1,2,3}
"""
import json
import mmap
import os

import numpy as np
import torch


class TwoBitLoader:
    """Load 2-bit packed experts; same interface as PackedExpertLoader."""

    def __init__(self, packed_dir, device=None):
        layout_path = os.path.join(packed_dir, 'layout.json')
        with open(layout_path) as f:
            L = json.load(f)

        self.packed_dir   = packed_dir
        self.device       = device or torch.device('cuda:0')
        self.group_size   = L['group_size']
        self.gu_shape     = tuple(L['gate_up_shape'])   # (2048, 4096)
        self.dn_shape     = tuple(L['down_shape'])      # (4096, 1024)
        self.gu_codes_bytes  = L['gu_codes_bytes']
        self.gu_scales_bytes = L['gu_scales_bytes']
        self.dn_codes_bytes  = L['dn_codes_bytes']
        self.dn_scales_bytes = L['dn_scales_bytes']
        self.expert_size  = L['expert_size']

        # Derived scale shapes
        gu_rows, gu_cols = self.gu_shape
        dn_rows, dn_cols = self.dn_shape
        gs = self.group_size
        self.gu_scales_shape = (gu_rows, gu_cols // gs)
        self.dn_scales_shape = (dn_rows, dn_cols // gs)

        self._mmaps = {}  # layer_idx -> (mmap, fd)

    def _ensure_mmap(self, layer):
        if layer not in self._mmaps:
            path = os.path.join(self.packed_dir, f'layer_{layer:02d}.bin')
            if not os.path.exists(path):
                raise FileNotFoundError(f'2-bit packed file not found: {path}')
            fd = os.open(path, os.O_RDONLY)
            sz = os.fstat(fd).st_size
            mm = mmap.mmap(fd, sz, access=mmap.ACCESS_READ)
            self._mmaps[layer] = (mm, fd)
        return self._mmaps[layer][0]

    def load_expert_raw(self, layer, expert_id):
        """Return raw bytes slice for one expert (zero-copy mmap view)."""
        mm = self._ensure_mmap(layer)
        off = expert_id * self.expert_size
        return mm[off:off + self.expert_size]

    def load_expert_raw_bytes(self, layer, expert_id):
        """Return a contiguous numpy uint8 copy (for CPU caching)."""
        raw = self.load_expert_raw(layer, expert_id)
        return np.frombuffer(raw, dtype=np.uint8).copy()

    @staticmethod
    def _dequant_on_gpu(codes_gpu, scales_gpu, rows, cols, group_size):
        """Vectorised 2-bit dequant entirely on GPU.

        codes_gpu:  uint8 tensor [rows*cols//4] — 4 codes per byte, LSB first
        scales_gpu: fp16 tensor [rows, cols//group_size]
        Returns: bf16 tensor [rows, cols]
        """
        n_groups = cols // group_size

        # Unpack 4 × 2-bit codes per byte → uint8 tensor [rows*cols]
        unpacked = torch.empty(codes_gpu.numel() * 4, dtype=torch.uint8, device=codes_gpu.device)
        unpacked[0::4] = codes_gpu & 0x3
        unpacked[1::4] = (codes_gpu >> 2) & 0x3
        unpacked[2::4] = (codes_gpu >> 4) & 0x3
        unpacked[3::4] = (codes_gpu >> 6) & 0x3

        # Reshape to [rows, n_groups, group_size]
        codes_3d = unpacked[:rows * cols].reshape(rows, n_groups, group_size)

        # Dequant: w = (code - 1.5) * scale
        scales_exp = scales_gpu.float().unsqueeze(-1)   # [rows, n_groups, 1]
        w = (codes_3d.float() - 1.5) * scales_exp       # [rows, n_groups, group_size]
        return w.reshape(rows, cols).to(torch.bfloat16)

    def dequant_raw_to_gpu(self, raw_bytes, device):
        """Transfer raw bytes to GPU then dequant. Returns (gate_up_w, down_w) bf16."""
        if isinstance(raw_bytes, torch.Tensor):
            buf_cpu = raw_bytes
        elif isinstance(raw_bytes, np.ndarray):
            buf_cpu = torch.from_numpy(raw_bytes)
        else:
            buf_cpu = torch.from_numpy(np.frombuffer(raw_bytes, dtype=np.uint8).copy())

        buf_gpu = buf_cpu.to(device, non_blocking=True)

        off = 0
        gu_codes  = buf_gpu[off:off + self.gu_codes_bytes].view(torch.uint8);  off += self.gu_codes_bytes
        gu_scales = buf_gpu[off:off + self.gu_scales_bytes].view(torch.float16).reshape(self.gu_scales_shape); off += self.gu_scales_bytes
        dn_codes  = buf_gpu[off:off + self.dn_codes_bytes].view(torch.uint8);  off += self.dn_codes_bytes
        dn_scales = buf_gpu[off:off + self.dn_scales_bytes].view(torch.float16).reshape(self.dn_scales_shape)

        gu_w = self._dequant_on_gpu(gu_codes, gu_scales, *self.gu_shape, self.group_size)
        dn_w = self._dequant_on_gpu(dn_codes, dn_scales, *self.dn_shape, self.group_size)
        return gu_w, dn_w

    def load_expert(self, layer, expert_id, device=None):
        """Load and dequant one expert. Returns (gate_up_w, down_w) bf16.

        gate_up_w: [2048, 4096] bf16  (used as weight in F.linear)
        down_w:    [4096, 1024] bf16
        """
        dev = device or self.device
        raw = self.load_expert_raw(layer, expert_id)
        return self.dequant_raw_to_gpu(raw, dev)

    def close(self):
        for mm, fd in self._mmaps.values():
            mm.close()
            os.close(fd)
        self._mmaps.clear()


def benchmark(packed_dir, layer=0, n_experts=20, device=None):
    import time
    dev = device or torch.device('cuda:0')
    loader = TwoBitLoader(packed_dir, dev)
    loader.load_expert(layer, 0, dev)  # warmup
    torch.cuda.synchronize(dev)

    times = []
    for i in range(n_experts):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        loader.load_expert(layer, i, dev)
        torch.cuda.synchronize(dev)
        times.append((time.perf_counter() - t0) * 1000)

    import numpy as np
    print(f'2-bit loader: mean={np.mean(times):.2f}ms  '
          f'p50={np.percentile(times,50):.2f}ms  '
          f'p99={np.percentile(times,99):.2f}ms  '
          f'8-expert={np.mean(times)*8:.1f}ms')
    loader.close()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/home/mh/models/Qwen3.5-397B-A17B/packed_experts_2bit')
    ap.add_argument('--layer', type=int, default=0)
    ap.add_argument('--n', type=int, default=20)
    args = ap.parse_args()
    benchmark(args.dir, layer=args.layer, n_experts=args.n)
