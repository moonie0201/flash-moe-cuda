#!/usr/bin/env python3
"""Repack experts from safetensors into contiguous per-layer binary files.

Creates packed_experts_bf16/layer_XX.bin where each expert is stored contiguously:
  [gate_up_proj bf16][down_proj bf16] per expert, 256 experts per layer.

Expert E data starts at offset: E * EXPERT_SIZE
Within each expert:
  [0 : gate_up_bytes]              gate_up_proj [1024, 2048] bf16
  [gate_up_bytes : expert_size]    down_proj [2048, 512] bf16

Usage:
    python ses/src/repack_packed.py --model-dir models/Qwen3.5-35B-A3B --output packed_experts_bf16
"""
import argparse
import json
import os
import sys
import time
import struct
import mmap as mmap_module

NUM_EXPERTS = 256
NUM_LAYERS = 40  # Qwen3.5-35B-A3B

# Expert component sizes (bf16)
GATE_UP_SHAPE = (1024, 2048)  # [intermediate*2, hidden]
DOWN_SHAPE = (2048, 512)       # [hidden, intermediate]
GATE_UP_BYTES = GATE_UP_SHAPE[0] * GATE_UP_SHAPE[1] * 2  # bf16 = 2 bytes
DOWN_BYTES = DOWN_SHAPE[0] * DOWN_SHAPE[1] * 2
EXPERT_SIZE = GATE_UP_BYTES + DOWN_BYTES  # 6,291,456 bytes = 6.0MB
LAYER_SIZE = NUM_EXPERTS * EXPERT_SIZE     # 1,610,612,736 bytes = 1.5GB


def load_safetensors_header(filepath):
    """Read safetensors header without loading data."""
    with open(filepath, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_size).decode('utf-8')
    return json.loads(header_json), 8 + header_size


def repack_layer(layer_idx, model_dir, index, output_dir):
    """Repack one layer's experts into a contiguous binary file."""
    weight_map = index['weight_map']

    gate_up_key = f'model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj'
    down_key = f'model.language_model.layers.{layer_idx}.mlp.experts.down_proj'

    if gate_up_key not in weight_map or down_key not in weight_map:
        print(f"  Layer {layer_idx}: keys not found, skipping")
        return 0

    out_path = os.path.join(output_dir, f'layer_{layer_idx:02d}.bin')
    t0 = time.monotonic()

    # Open output file
    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, LAYER_SIZE)

    # Load gate_up_proj shard via mmap
    gu_shard = os.path.join(model_dir, weight_map[gate_up_key])
    gu_header, gu_data_offset = load_safetensors_header(gu_shard)
    gu_meta = gu_header[gate_up_key]
    gu_start, gu_end = gu_meta['data_offsets']

    # Load down_proj shard via mmap
    dw_shard = os.path.join(model_dir, weight_map[down_key])
    dw_header, dw_data_offset = load_safetensors_header(dw_shard)
    dw_meta = dw_header[down_key]
    dw_start, dw_end = dw_meta['data_offsets']

    # mmap source files
    gu_fd = os.open(gu_shard, os.O_RDONLY)
    dw_fd = os.open(dw_shard, os.O_RDONLY)

    gu_file_size = os.fstat(gu_fd).st_size
    dw_file_size = os.fstat(dw_fd).st_size

    gu_mm = mmap_module.mmap(gu_fd, gu_file_size, access=mmap_module.ACCESS_READ)
    dw_mm = mmap_module.mmap(dw_fd, dw_file_size, access=mmap_module.ACCESS_READ)

    # Per-expert size in source tensor
    gu_expert_bytes = GATE_UP_SHAPE[0] * GATE_UP_SHAPE[1] * 2  # bf16
    dw_expert_bytes = DOWN_SHAPE[0] * DOWN_SHAPE[1] * 2

    bytes_written = 0
    try:
        for expert_idx in range(NUM_EXPERTS):
            gu_src_offset = gu_data_offset + gu_start + expert_idx * gu_expert_bytes
            gu_data = gu_mm[gu_src_offset:gu_src_offset + gu_expert_bytes]

            dw_src_offset = dw_data_offset + dw_start + expert_idx * dw_expert_bytes
            dw_data = dw_mm[dw_src_offset:dw_src_offset + dw_expert_bytes]

            dst_offset = expert_idx * EXPERT_SIZE
            os.pwrite(fd_out, gu_data, dst_offset)
            os.pwrite(fd_out, dw_data, dst_offset + GATE_UP_BYTES)
            bytes_written += EXPERT_SIZE
    finally:
        gu_mm.close()
        dw_mm.close()
        os.close(gu_fd)
        os.close(dw_fd)
        os.close(fd_out)

    elapsed = time.monotonic() - t0
    throughput = bytes_written / elapsed / (1024**3) if elapsed > 0 else 0
    print(f"  Layer {layer_idx:2d}: {bytes_written/1e9:.2f}GB in {elapsed:.1f}s ({throughput:.1f} GB/s)")
    return bytes_written


def write_layout(output_dir):
    """Write layout.json for the packed format."""
    layout = {
        "format": "packed_bf16",
        "expert_size": EXPERT_SIZE,
        "num_experts": NUM_EXPERTS,
        "num_layers": NUM_LAYERS,
        "gate_up_shape": list(GATE_UP_SHAPE),
        "down_shape": list(DOWN_SHAPE),
        "gate_up_bytes": GATE_UP_BYTES,
        "down_bytes": DOWN_BYTES,
        "gate_up_offset": 0,
        "down_offset": GATE_UP_BYTES,
    }
    with open(os.path.join(output_dir, 'layout.json'), 'w') as f:
        json.dump(layout, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', default='packed_experts_bf16')
    parser.add_argument('--layers', default=None, help='e.g. "0,10,20" or "0-5"')
    args = parser.parse_args()

    index_path = os.path.join(args.model_dir, 'model.safetensors.index.json')
    with open(index_path) as f:
        index = json.load(f)

    os.makedirs(args.output, exist_ok=True)
    write_layout(args.output)

    if args.layers:
        layer_list = []
        for part in args.layers.split(','):
            if '-' in part:
                a, b = part.split('-')
                layer_list.extend(range(int(a), int(b)+1))
            else:
                layer_list.append(int(part))
    else:
        layer_list = list(range(NUM_LAYERS))

    total_gb = len(layer_list) * LAYER_SIZE / 1e9
    print(f"Repacking {len(layer_list)} layers ({total_gb:.1f}GB) to {args.output}/")

    t_start = time.monotonic()
    total_bytes = 0
    for layer_idx in layer_list:
        total_bytes += repack_layer(layer_idx, args.model_dir, index, args.output)

    elapsed = time.monotonic() - t_start
    print(f"\nDone: {total_bytes/1e9:.1f}GB in {elapsed:.0f}s "
          f"({total_bytes/elapsed/1e9:.1f} GB/s)")


if __name__ == '__main__':
    main()
