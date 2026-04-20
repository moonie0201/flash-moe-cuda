#!/usr/bin/env python3
"""Repack GPTQ-Int4 397B experts into contiguous per-layer binary files.

Each expert's 12 tensors (3 projections × 4 tensors each) are packed into
a single contiguous byte block. This enables a single CPU→GPU transfer
instead of 12 small transfers.

Layout per expert (6,291,520 bytes = 6MB):
  gate_proj: qweight(2MB) + qzeros(16KB) + scales(64KB) + g_idx(16KB) = 2,162,688
  up_proj:   same = 2,162,688
  down_proj: qweight(2MB) + qzeros(16KB) + scales(64KB) + g_idx(4KB)  = 2,117,632

Per layer: 512 × 6,442,008 bytes ≈ 3.1 GB
Total 60 layers: ~186GB

Usage:
    python ses/src/repack_397b_gptq.py --layers 0-9
    python ses/src/repack_397b_gptq.py --resume
"""
import argparse
import json
import os
import struct
import sys
import time

MODEL_DIR = '/home/mh/models/Qwen3.5-397B-A17B-GPTQ-Int4'
OUTPUT_DIR = '/home/mh/models/Qwen3.5-397B-A17B-GPTQ-Int4/packed_experts'

NUM_EXPERTS = 512
NUM_LAYERS = 60

# Component sizes per projection (bytes)
# gate_proj/up_proj: in=4096, out=1024, groups=32
GATE_UP_QW_BYTES  = 512 * 1024 * 4       # [512, 1024] I32 = 2,097,152
GATE_UP_QZ_BYTES  = 32 * 128 * 4         # [32, 128] I32 = 16,384
GATE_UP_SC_BYTES  = 32 * 1024 * 2        # [32, 1024] F16 = 65,536
GATE_UP_GI_BYTES  = 4096 * 4             # [4096] I32 = 16,384
GATE_UP_PROJ_BYTES = GATE_UP_QW_BYTES + GATE_UP_QZ_BYTES + GATE_UP_SC_BYTES + GATE_UP_GI_BYTES

# down_proj: in=1024, out=4096, groups=8
DOWN_QW_BYTES = 128 * 4096 * 4           # [128, 4096] I32 = 2,097,152
DOWN_QZ_BYTES = 8 * 512 * 4              # [8, 512] I32 = 16,384
DOWN_SC_BYTES = 8 * 4096 * 2             # [8, 4096] F16 = 65,536
DOWN_GI_BYTES = 1024 * 4                 # [1024] I32 = 4,096
DOWN_PROJ_BYTES = DOWN_QW_BYTES + DOWN_QZ_BYTES + DOWN_SC_BYTES + DOWN_GI_BYTES

EXPERT_SIZE = 2 * GATE_UP_PROJ_BYTES + DOWN_PROJ_BYTES
LAYER_SIZE = NUM_EXPERTS * EXPERT_SIZE

# Offsets within expert block (for the loader)
OFFSETS = {
    'gate_proj.qweight':  (0, GATE_UP_QW_BYTES),
    'gate_proj.qzeros':   (GATE_UP_QW_BYTES, GATE_UP_QZ_BYTES),
    'gate_proj.scales':   (GATE_UP_QW_BYTES + GATE_UP_QZ_BYTES, GATE_UP_SC_BYTES),
    'gate_proj.g_idx':    (GATE_UP_QW_BYTES + GATE_UP_QZ_BYTES + GATE_UP_SC_BYTES, GATE_UP_GI_BYTES),
    'up_proj.qweight':    (GATE_UP_PROJ_BYTES, GATE_UP_QW_BYTES),
    'up_proj.qzeros':     (GATE_UP_PROJ_BYTES + GATE_UP_QW_BYTES, GATE_UP_QZ_BYTES),
    'up_proj.scales':     (GATE_UP_PROJ_BYTES + GATE_UP_QW_BYTES + GATE_UP_QZ_BYTES, GATE_UP_SC_BYTES),
    'up_proj.g_idx':      (GATE_UP_PROJ_BYTES + GATE_UP_QW_BYTES + GATE_UP_QZ_BYTES + GATE_UP_SC_BYTES, GATE_UP_GI_BYTES),
    'down_proj.qweight':  (2 * GATE_UP_PROJ_BYTES, DOWN_QW_BYTES),
    'down_proj.qzeros':   (2 * GATE_UP_PROJ_BYTES + DOWN_QW_BYTES, DOWN_QZ_BYTES),
    'down_proj.scales':   (2 * GATE_UP_PROJ_BYTES + DOWN_QW_BYTES + DOWN_QZ_BYTES, DOWN_SC_BYTES),
    'down_proj.g_idx':    (2 * GATE_UP_PROJ_BYTES + DOWN_QW_BYTES + DOWN_QZ_BYTES + DOWN_SC_BYTES, DOWN_GI_BYTES),
}


def safetensors_header(path):
    with open(path, 'rb') as f:
        h = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(h))
    return 8 + h, hdr


def repack_layer(layer_idx, weight_map, shard_cache, output_dir):
    """Repack one layer's 512 experts into a contiguous binary file."""
    out_path = os.path.join(output_dir, f'layer_{layer_idx:02d}.bin')
    t0 = time.monotonic()

    def get_shard(fname):
        if fname not in shard_cache:
            path = os.path.join(MODEL_DIR, fname)
            do, hdr = safetensors_header(path)
            fd = os.open(path, os.O_RDONLY)
            shard_cache[fname] = (fd, do, hdr)
        return shard_cache[fname]

    # Allocate output
    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, LAYER_SIZE)

    for e in range(NUM_EXPERTS):
        prefix = f'model.language_model.layers.{layer_idx}.mlp.experts.{e}'
        base_dst = e * EXPERT_SIZE

        for comp, (off_in_exp, size) in OFFSETS.items():
            key = f'{prefix}.{comp}'
            shard_name = weight_map[key]
            fd, do, hdr = get_shard(shard_name)
            meta = hdr[key]
            start, end = meta['data_offsets']
            actual_size = end - start
            if actual_size != size:
                raise ValueError(f'{key}: size {actual_size} != expected {size}')
            data = os.pread(fd, size, do + start)
            os.pwrite(fd_out, data, base_dst + off_in_exp)

    os.close(fd_out)
    return time.monotonic() - t0


def write_layout(output_dir):
    layout = {
        'model': 'Qwen3.5-397B-A17B-GPTQ-Int4',
        'num_layers': NUM_LAYERS,
        'num_experts': NUM_EXPERTS,
        'expert_size': EXPERT_SIZE,
        'offsets': {k: {'offset': v[0], 'size': v[1]} for k, v in OFFSETS.items()},
    }
    with open(os.path.join(output_dir, 'layout.json'), 'w') as f:
        json.dump(layout, f, indent=2)


def parse_layers(spec):
    if not spec:
        return list(range(NUM_LAYERS))
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', default=None)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_layout(OUTPUT_DIR)

    print('Loading weight index...')
    with open(os.path.join(MODEL_DIR, 'model.safetensors.index.json')) as f:
        weight_map = json.load(f)['weight_map']

    layers = parse_layers(args.layers)
    if args.resume:
        layers = [l for l in layers
                  if not os.path.exists(os.path.join(OUTPUT_DIR, f'layer_{l:02d}.bin'))]

    total_gb = len(layers) * LAYER_SIZE / 1024**3
    print(f'Layers: {len(layers)}, output: ~{total_gb:.1f} GB')
    print(f'Expert size: {EXPERT_SIZE:,} bytes ({EXPERT_SIZE/1024/1024:.2f} MB)')

    shard_cache = {}
    t_start = time.monotonic()
    for i, layer_idx in enumerate(layers):
        dt = repack_layer(layer_idx, weight_map, shard_cache, OUTPUT_DIR)
        overall = time.monotonic() - t_start
        eta = (len(layers) - i - 1) * overall / (i + 1)
        print(f'L{layer_idx:2d}: {LAYER_SIZE/1024**3:.2f}GB in {dt:.1f}s '
              f'({LAYER_SIZE/dt/1024**3:.2f} GB/s) ETA {eta:.0f}s', flush=True)

    for fd, _, _ in shard_cache.values():
        os.close(fd)


if __name__ == '__main__':
    main()
