#!/usr/bin/env python3
"""Repack 397B expert weights from safetensors into contiguous per-layer binary files.

Output: packed_experts_bf16/layer_XX.bin
Per expert: [gate_up_proj bf16][down_proj bf16]
  gate_up: [2048, 4096] = 16,777,216 bytes
  down:    [4096, 1024] =  8,388,608 bytes
  total:                  25,165,824 bytes = 24MB per expert

Per layer: 512 experts × 24MB = 12,288MB (~12GB)
Total: 60 layers × 12GB = ~720GB

Usage:
    python ses/src/repack_397b.py
    python ses/src/repack_397b.py --layers 0-9
    python ses/src/repack_397b.py --resume   # skip already-done layers
"""
import argparse
import json
import os
import struct
import sys
import time

MODEL_DIR = '/home/mh/models/Qwen3.5-397B-A17B'
OUTPUT_DIR = '/home/mh/models/Qwen3.5-397B-A17B/packed_experts_bf16'

NUM_EXPERTS = 512
NUM_LAYERS = 60
GATE_UP_BYTES = 2048 * 4096 * 2   # 16,777,216
DOWN_BYTES    = 4096 * 1024 * 2   #  8,388,608
EXPERT_SIZE   = GATE_UP_BYTES + DOWN_BYTES  # 25,165,824
LAYER_SIZE    = NUM_EXPERTS * EXPERT_SIZE   # 12,884,901,888 (~12GB)


def load_index():
    with open(os.path.join(MODEL_DIR, 'model.safetensors.index.json')) as f:
        return json.load(f)['weight_map']


def safetensors_data_offset(filepath):
    with open(filepath, 'rb') as f:
        h = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(h))
    return 8 + h, hdr


def repack_layer(layer_idx, weight_map, shard_cache, output_dir):
    gu_key = f'model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj'
    dn_key = f'model.language_model.layers.{layer_idx}.mlp.experts.down_proj'

    gu_file = weight_map[gu_key]
    dn_file = weight_map[dn_key]

    out_path = os.path.join(output_dir, f'layer_{layer_idx:02d}.bin')
    t0 = time.monotonic()

    # Get shard offsets/headers (cached)
    def get_shard(fname):
        if fname not in shard_cache:
            path = os.path.join(MODEL_DIR, fname)
            data_off, hdr = safetensors_data_offset(path)
            fd = os.open(path, os.O_RDONLY)
            shard_cache[fname] = (fd, data_off, hdr)
        return shard_cache[fname]

    gu_fd, gu_base, gu_hdr = get_shard(gu_file)
    dn_fd, dn_base, dn_hdr = get_shard(dn_file)

    gu_info = gu_hdr[gu_key]
    dn_info = dn_hdr[dn_key]

    # Absolute byte offsets in shard files
    gu_start = gu_base + gu_info['data_offsets'][0]
    dn_start = dn_base + dn_info['data_offsets'][0]

    expected_gu = NUM_EXPERTS * GATE_UP_BYTES
    expected_dn = NUM_EXPERTS * DOWN_BYTES
    actual_gu = gu_info['data_offsets'][1] - gu_info['data_offsets'][0]
    actual_dn = dn_info['data_offsets'][1] - dn_info['data_offsets'][0]
    if actual_gu != expected_gu:
        raise ValueError(f'Layer {layer_idx} gate_up size mismatch: {actual_gu} vs {expected_gu}')
    if actual_dn != expected_dn:
        raise ValueError(f'Layer {layer_idx} down size mismatch: {actual_dn} vs {expected_dn}')

    # Pre-allocate output
    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, LAYER_SIZE)

    # Write per expert: gate_up then down
    for e in range(NUM_EXPERTS):
        dst = e * EXPERT_SIZE
        gu_data = os.pread(gu_fd, GATE_UP_BYTES, gu_start + e * GATE_UP_BYTES)
        os.pwrite(fd_out, gu_data, dst)
        dn_data = os.pread(dn_fd, DOWN_BYTES, dn_start + e * DOWN_BYTES)
        os.pwrite(fd_out, dn_data, dst + GATE_UP_BYTES)

    os.close(fd_out)
    return time.monotonic() - t0


def write_layout(output_dir):
    layout = {
        'model': 'Qwen3.5-397B-A17B',
        'num_layers': NUM_LAYERS,
        'num_experts': NUM_EXPERTS,
        'expert_size': EXPERT_SIZE,
        'gate_up_bytes': GATE_UP_BYTES,
        'down_bytes': DOWN_BYTES,
        'gate_up_shape': [2048, 4096],
        'down_shape': [4096, 1024],
        'dtype': 'bf16',
    }
    with open(os.path.join(output_dir, 'layout.json'), 'w') as f:
        json.dump(layout, f, indent=2)


def parse_layers(spec):
    if not spec:
        return list(range(NUM_LAYERS))
    layers = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', default=None)
    parser.add_argument('--resume', action='store_true', help='Skip already-completed layers')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_layout(OUTPUT_DIR)

    print('Loading weight index...')
    weight_map = load_index()

    layers = parse_layers(args.layers)

    if args.resume:
        layers = [l for l in layers
                  if not os.path.exists(os.path.join(OUTPUT_DIR, f'layer_{l:02d}.bin'))]
        print(f'Resume: {len(layers)} layers remaining')

    total_gb = len(layers) * LAYER_SIZE / 1024**3
    stat = os.statvfs(OUTPUT_DIR)
    free_gb = stat.f_bavail * stat.f_frsize / 1024**3
    print(f'Layers: {len(layers)}, output: ~{total_gb:.1f} GB, free: {free_gb:.1f} GB')
    if free_gb < total_gb + 10:
        print(f'WARNING: low disk space')
        sys.exit(1)

    shard_cache = {}
    t_start = time.monotonic()
    total_written = 0

    for i, layer_idx in enumerate(layers):
        elapsed = repack_layer(layer_idx, weight_map, shard_cache, OUTPUT_DIR)
        total_written += LAYER_SIZE
        overall = time.monotonic() - t_start
        throughput = LAYER_SIZE / elapsed / 1024**3
        avg = total_written / overall / 1024**3
        eta = (len(layers) - i - 1) * (overall / (i + 1))
        print(f'Layer {layer_idx:2d}: {LAYER_SIZE/1024**3:.1f}GB in {elapsed:.1f}s '
              f'({throughput:.2f} GB/s) | avg {avg:.2f} GB/s | ETA {eta:.0f}s '
              f'[{i+1}/{len(layers)}]', flush=True)

    for fd, _, _ in shard_cache.values():
        os.close(fd)

    total_elapsed = time.monotonic() - t_start
    print(f'\nDone: {total_written/1024**3:.1f} GB in {total_elapsed:.0f}s '
          f'({total_written/total_elapsed/1024**3:.2f} GB/s avg)')


if __name__ == '__main__':
    main()
