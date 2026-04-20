#!/usr/bin/env python3
"""Analyze real expert weights from Qwen3.5-35B-A3B for SVD quality.

This is Experiment 1 from the SES paper: validate that expert weights
have low-rank structure suitable for hierarchical decomposition.

Usage:
    python ses/src/analyze_real_experts.py --model-dir models/Qwen3.5-35B-A3B
"""
import argparse
import json
import numpy as np
import os
import sys
import time

try:
    from safetensors import safe_open
except ImportError:
    print("pip install safetensors required")
    sys.exit(1)

from svd_decompose import svd_decompose, expert_forward, base_expert_forward


def load_expert_weights(model_dir: str, layer: int, expert_id: int):
    """Load one expert's weights from safetensors files.

    Returns dict with 'gate_up_proj' and 'down_proj' as numpy arrays.
    """
    index_path = os.path.join(model_dir, 'model.safetensors.index.json')
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index['weight_map']

    # Expert weights are stored as packed tensors:
    # layers.{L}.mlp.experts.gate_up_proj -> [num_experts, 2*intermediate, hidden]
    # layers.{L}.mlp.experts.down_proj -> [num_experts, hidden, intermediate]
    prefix = f"model.language_model.layers.{layer}.mlp.experts"

    result = {}
    for key_suffix in ['gate_up_proj', 'down_proj']:
        key = f"{prefix}.{key_suffix}"
        if key not in weight_map:
            print(f"Warning: {key} not found in weight map")
            continue

        filename = weight_map[key]
        filepath = os.path.join(model_dir, filename)

        with safe_open(filepath, framework="numpy") as f:
            tensor = f.get_tensor(key)
            # Shape: [num_experts, out_dim, in_dim]
            print(f"  {key_suffix}: shape={tensor.shape}, dtype={tensor.dtype}")

            if tensor.ndim == 3:
                expert_weight = tensor[expert_id]  # [out_dim, in_dim]
            else:
                expert_weight = tensor

            result[key_suffix] = expert_weight.astype(np.float32)

    return result


def analyze_expert_svd(weights: dict, ranks=(8, 16, 32, 64, 128)):
    """Run SVD analysis on one expert's projections."""
    results = {}

    for proj_name, W in weights.items():
        if proj_name == 'gate_up_proj':
            # Split gate_up into gate and up (first half = gate, second = up)
            mid = W.shape[0] // 2
            projections = {
                'gate_proj': W[:mid, :],
                'up_proj': W[mid:, :],
            }
        else:
            projections = {proj_name: W}

        for name, matrix in projections.items():
            print(f"\n  {name}: shape={matrix.shape}")
            proj_results = {}

            # Full SVD for singular value spectrum
            _, S_full, _ = np.linalg.svd(matrix, full_matrices=False)
            proj_results['singular_values'] = S_full[:128].tolist()
            proj_results['sv_ratio_top64'] = float(
                np.sum(S_full[:64] ** 2) / np.sum(S_full ** 2)
            )

            # Per-rank analysis
            x = np.random.randn(matrix.shape[1]).astype(np.float32)
            x = x / np.linalg.norm(x)
            full_out = matrix @ x

            for rank in ranks:
                if rank > min(matrix.shape):
                    continue
                base, residual, metrics = svd_decompose(matrix, rank)
                base_out = base @ x
                cos_sim = float(np.dot(full_out, base_out) / (
                    np.linalg.norm(full_out) * np.linalg.norm(base_out) + 1e-10
                ))

                base_bytes = rank * (matrix.shape[0] + matrix.shape[1]) * 4
                orig_bytes = matrix.shape[0] * matrix.shape[1] * 4
                compression = base_bytes / orig_bytes

                proj_results[f'rank_{rank}'] = {
                    'energy_ratio': metrics['energy_ratio'],
                    'relative_error': metrics['relative_error'],
                    'cosine_sim': cos_sim,
                    'compression_ratio': compression,
                }

                print(f"    rank-{rank:3d}: energy={metrics['energy_ratio']:.4f} "
                      f"error={metrics['relative_error']:.4f} "
                      f"cos_sim={cos_sim:.6f} "
                      f"compress={compression:.3f}")

            results[name] = proj_results

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--layers', default='0,10,20,30,39',
                        help='Comma-separated layer indices')
    parser.add_argument('--experts-per-layer', type=int, default=5)
    parser.add_argument('--output', default='experiments/svd_real_results.json')
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(',')]
    all_results = {}

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        # Sample random experts
        np.random.seed(layer_idx)
        expert_ids = np.random.choice(256, args.experts_per_layer, replace=False)

        layer_results = {}
        for expert_id in expert_ids:
            print(f"\n  Expert {expert_id}:")
            try:
                weights = load_expert_weights(args.model_dir, layer_idx, expert_id)
                result = analyze_expert_svd(weights)
                layer_results[int(expert_id)] = result
            except Exception as e:
                print(f"    ERROR: {e}")
                continue

        all_results[layer_idx] = layer_results

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for rank in [32, 64, 128]:
        sims = []
        energies = []
        for layer_data in all_results.values():
            for expert_data in layer_data.values():
                for proj_data in expert_data.values():
                    key = f'rank_{rank}'
                    if key in proj_data:
                        sims.append(proj_data[key]['cosine_sim'])
                        energies.append(proj_data[key]['energy_ratio'])
        if sims:
            print(f"  rank-{rank}: cos_sim={np.mean(sims):.4f}±{np.std(sims):.4f} "
                  f"energy={np.mean(energies):.4f}±{np.std(energies):.4f} "
                  f"(n={len(sims)})")


if __name__ == '__main__':
    main()
