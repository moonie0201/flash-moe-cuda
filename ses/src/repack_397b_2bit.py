#!/usr/bin/env python3
"""Repack 397B bf16 expert weights to 2-bit with group-wise scale.

Input:  packed_experts_bf16/layer_XX.bin  (24MB/expert, 512 experts/layer, 60 layers)
Output: packed_experts_2bit/layer_XX.bin  (3.54MB/expert, 108.7GB total)

2-bit symmetric quantization with group_size=64:
  4 levels: codes {0,1,2,3} → {-1.5s, -0.5s, +0.5s, +1.5s}  where s = max|w|/1.5
  Pack 4 codes per byte (LSB first)

Layout per expert (3,538,944 bytes):
  gate_up_codes:  2,097,152 bytes  [2048*4096 / 4]
  gate_up_scales:   262,144 bytes  [2048 * 64 * 2]   fp16, shape [2048, 64]
  down_codes:     1,048,576 bytes  [4096*1024 / 4]
  down_scales:      131,072 bytes  [4096 * 16 * 2]   fp16, shape [4096, 16]

Per-layer: 512 × 3.54MB ≈ 1.81GB
Total:      60 × 1.81GB ≈ 108.7GB  (vs 720GB bf16)

GPU-batched: processes BATCH experts at once on CUDA for 10-20× speedup.

Usage:
    python ses/src/repack_397b_2bit.py
    python ses/src/repack_397b_2bit.py --layers 0-9
    python ses/src/repack_397b_2bit.py --resume
    python ses/src/repack_397b_2bit.py --device cuda:1  # use second GPU
"""
import argparse
import json
import mmap
import os
import sys
import time

import numpy as np
import torch

INPUT_DIR  = '/home/mh/models/Qwen3.5-397B-A17B/packed_experts_bf16'
OUTPUT_DIR = '/home/mh/models/Qwen3.5-397B-A17B/packed_experts_2bit'

NUM_EXPERTS  = 512
NUM_LAYERS   = 60
GROUP_SIZE   = 64

GATE_UP_ROWS, GATE_UP_COLS = 2048, 4096
DOWN_ROWS,    DOWN_COLS    = 4096, 1024

GATE_UP_BYTES     = GATE_UP_ROWS * GATE_UP_COLS * 2  # bf16
DOWN_BYTES        = DOWN_ROWS    * DOWN_COLS    * 2
EXPERT_SIZE_BF16  = GATE_UP_BYTES + DOWN_BYTES   # 25,165,824

GU_CODES_BYTES  = GATE_UP_ROWS * GATE_UP_COLS // 4   # 2,097,152
GU_SCALES_BYTES = GATE_UP_ROWS * (GATE_UP_COLS // GROUP_SIZE) * 2  # 262,144
DN_CODES_BYTES  = DOWN_ROWS    * DOWN_COLS    // 4   # 1,048,576
DN_SCALES_BYTES = DOWN_ROWS    * (DOWN_COLS   // GROUP_SIZE) * 2   # 131,072
EXPERT_SIZE_2BIT = GU_CODES_BYTES + GU_SCALES_BYTES + DN_CODES_BYTES + DN_SCALES_BYTES  # 3,538,944


def quantize_2bit_gpu(w_bf16, group_size=64):
    """Quantize a batch of [N, rows, cols] bf16 tensors on GPU.

    Returns:
        codes_packed: uint8 tensor [N, rows*cols//4]
        scales:       fp16 tensor  [N, rows, cols//group_size]
    """
    N, rows, cols = w_bf16.shape
    n_groups = cols // group_size

    w = w_bf16.float().reshape(N, rows, n_groups, group_size)  # [N, rows, n_groups, gs]

    # Scale: max|w|/1.5 per group
    abs_max = w.abs().amax(dim=-1)                   # [N, rows, n_groups]
    scales = (abs_max / 1.5).to(torch.float16)       # store as fp16

    # Quantize: code = round(w/s + 1.5), clamp [0, 3]
    s_safe = scales.float().clamp(min=1e-30).unsqueeze(-1)  # [N, rows, n_groups, 1]
    codes = (w / s_safe + 1.5).round().clamp(0, 3).to(torch.uint8)  # [N, rows, n_groups, gs]

    # Pack 4 codes per byte, LSB first
    c = codes.reshape(N, -1)  # [N, rows*n_groups*gs]
    packed = (c[:, 0::4]
              | (c[:, 1::4] << 2)
              | (c[:, 2::4] << 4)
              | (c[:, 3::4] << 6)).to(torch.uint8)  # [N, rows*cols//4]

    return packed, scales


def repack_layer(layer_idx, out_dir, device, batch_size=32):
    in_path  = os.path.join(INPUT_DIR,  f'layer_{layer_idx:02d}.bin')
    out_path = os.path.join(out_dir, f'layer_{layer_idx:02d}.bin')

    t0 = time.monotonic()
    fd_in = os.open(in_path, os.O_RDONLY)
    mm = mmap.mmap(fd_in, 0, access=mmap.ACCESS_READ)

    # Read all bf16 data as a big numpy array
    raw = np.frombuffer(mm, dtype=np.uint16).copy()  # [512 * EXPERT_SIZE_BF16 / 2]
    mm.close()
    os.close(fd_in)

    experts_raw = raw.reshape(NUM_EXPERTS, EXPERT_SIZE_BF16 // 2)  # [512, elements_per_expert]
    gu_end = GATE_UP_BYTES // 2  # index in uint16 array

    out_buf = np.empty(NUM_EXPERTS * EXPERT_SIZE_2BIT, dtype=np.uint8)

    # Process in GPU batches
    for b_start in range(0, NUM_EXPERTS, batch_size):
        b_end = min(b_start + batch_size, NUM_EXPERTS)
        B = b_end - b_start

        # Gate+Up: [B, 2048, 4096] bf16
        gu_u16 = experts_raw[b_start:b_end, :gu_end]              # [B, 2048*4096]
        gu_t = torch.from_numpy(gu_u16.view(np.int16)).view(torch.bfloat16)
        gu_t = gu_t.reshape(B, GATE_UP_ROWS, GATE_UP_COLS).to(device)

        # Down: [B, 4096, 1024] bf16
        dn_u16 = experts_raw[b_start:b_end, gu_end:]              # [B, 4096*1024]
        dn_t = torch.from_numpy(dn_u16.view(np.int16)).view(torch.bfloat16)
        dn_t = dn_t.reshape(B, DOWN_ROWS, DOWN_COLS).to(device)

        gu_codes, gu_scales = quantize_2bit_gpu(gu_t, GROUP_SIZE)  # [B, ...]
        dn_codes, dn_scales = quantize_2bit_gpu(dn_t, GROUP_SIZE)

        # Move back to CPU
        gu_codes_np  = gu_codes.cpu().numpy()   # [B, GU_CODES_BYTES]
        gu_scales_np = gu_scales.cpu().numpy()  # [B, 2048, 64] fp16
        dn_codes_np  = dn_codes.cpu().numpy()   # [B, DN_CODES_BYTES]
        dn_scales_np = dn_scales.cpu().numpy()  # [B, 4096, 16] fp16

        # Write each expert into output buffer
        for i in range(B):
            eid = b_start + i
            off = eid * EXPERT_SIZE_2BIT
            out_buf[off:off + GU_CODES_BYTES]  = gu_codes_np[i].view(np.uint8)
            o2 = off + GU_CODES_BYTES
            out_buf[o2:o2 + GU_SCALES_BYTES]  = gu_scales_np[i].reshape(-1).view(np.uint8)
            o3 = o2 + GU_SCALES_BYTES
            out_buf[o3:o3 + DN_CODES_BYTES]   = dn_codes_np[i].view(np.uint8)
            o4 = o3 + DN_CODES_BYTES
            out_buf[o4:o4 + DN_SCALES_BYTES]  = dn_scales_np[i].reshape(-1).view(np.uint8)

    with open(out_path, 'wb') as f:
        f.write(out_buf.tobytes())

    elapsed = time.monotonic() - t0
    in_gb  = (NUM_EXPERTS * EXPERT_SIZE_BF16) / 1e9
    out_mb = (NUM_EXPERTS * EXPERT_SIZE_2BIT) / 1e6
    print(f'  layer {layer_idx:02d}: {in_gb:.1f}GB → {out_mb:.0f}MB  {elapsed:.1f}s')


def write_layout(out_dir):
    layout = {
        'num_experts':      NUM_EXPERTS,
        'num_layers':       NUM_LAYERS,
        'group_size':       GROUP_SIZE,
        'gate_up_shape':    [GATE_UP_ROWS, GATE_UP_COLS],
        'down_shape':       [DOWN_ROWS, DOWN_COLS],
        'gu_codes_bytes':   GU_CODES_BYTES,
        'gu_scales_bytes':  GU_SCALES_BYTES,
        'dn_codes_bytes':   DN_CODES_BYTES,
        'dn_scales_bytes':  DN_SCALES_BYTES,
        'expert_size':      EXPERT_SIZE_2BIT,
    }
    with open(os.path.join(out_dir, 'layout.json'), 'w') as f:
        json.dump(layout, f, indent=2)
    print(f'Wrote layout.json: {EXPERT_SIZE_2BIT} bytes/expert '
          f'({EXPERT_SIZE_2BIT/1e6:.2f}MB), '
          f'{NUM_EXPERTS*EXPERT_SIZE_2BIT/1e6:.0f}MB/layer, '
          f'{NUM_LAYERS*NUM_EXPERTS*EXPERT_SIZE_2BIT/1e9:.1f}GB total')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', default=None,
                    help='Layer range, e.g. "0-9" or "5"')
    ap.add_argument('--resume', action='store_true',
                    help='Skip layers whose output file already exists')
    ap.add_argument('--out-dir', default=OUTPUT_DIR)
    ap.add_argument('--device', default='cuda:0',
                    help='GPU device for quantization (default: cuda:0)')
    ap.add_argument('--batch', type=int, default=32,
                    help='Experts per GPU batch (default: 32, reduce if OOM)')
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.layers:
        parts = args.layers.split('-')
        start, end = int(parts[0]), int(parts[-1])
        layers = list(range(start, end + 1))
    else:
        layers = list(range(NUM_LAYERS))

    write_layout(out_dir)

    total_start = time.monotonic()
    done = 0
    for layer_idx in layers:
        out_path = os.path.join(out_dir, f'layer_{layer_idx:02d}.bin')
        if args.resume and os.path.exists(out_path):
            sz = os.path.getsize(out_path)
            if sz == NUM_EXPERTS * EXPERT_SIZE_2BIT:
                print(f'  layer {layer_idx:02d}: skip (exists)')
                done += 1
                continue
        print(f'layer {layer_idx:02d} ...', flush=True)
        repack_layer(layer_idx, out_dir, device, args.batch)
        done += 1
        elapsed = time.monotonic() - total_start
        remaining = len(layers) - done
        eta = (elapsed / done) * remaining
        print(f'  Progress: {done}/{len(layers)} layers, eta {eta/60:.1f}min', flush=True)

    print(f'\nDone: {len(layers)} layers in {(time.monotonic()-total_start)/60:.1f}min')


if __name__ == '__main__':
    main()
