#!/usr/bin/env python3
"""Generate paper figures from experiment results."""
import matplotlib.pyplot as plt
import numpy as np
import json
import os

plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

OUTPUT_DIR = 'experiments/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig_cache_hit_vs_size():
    """Fig 3: Cache hit rate vs HOT cache percentage."""
    # Data from multi-layer benchmark
    hot_pcts = [20, 30, 50]
    layers_data = {
        'L0':  [34.2, 44.5, 69.2],
        'L10': [42.8, 59.5, 81.2],
        'L20': [54.9, 64.5, 80.8],
        'L30': [41.8, 57.5, 75.5],
        'L39': [46.4, 60.5, 84.0],
    }

    fig, ax = plt.subplots()
    for label, hits in layers_data.items():
        ax.plot(hot_pcts, hits, 'o-', label=label, linewidth=2, markersize=8)

    ax.set_xlabel('HOT Cache Size (% of experts)')
    ax.set_ylabel('Cache Hit Rate (%)')
    ax.set_title('Expert Cache Hit Rate vs Cache Size')
    ax.legend()
    ax.set_ylim(20, 90)
    ax.set_xticks(hot_pcts)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig3_cache_hit_vs_size.png')
    print(f'Saved fig3_cache_hit_vs_size.png')
    plt.close()


def fig_throughput_comparison():
    """Fig 4: Throughput comparison bar chart."""
    systems = [
        'Mac M3 Max\n(baseline)',
        'PC\nno cache',
        'PC\nHOT 20%',
        'PC\nHOT 30%',
        'PC\nHOT 50%',
    ]
    toks = [4.36, 2.15, 3.10, 3.68, 5.13]
    colors = ['#888888', '#cc4444', '#ee8844', '#88bb44', '#2288cc']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(systems, toks, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, toks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Tokens per second')
    ax.set_title('MoE Inference Throughput: Mac vs PC with Frequency-Aware Caching')
    ax.axhline(y=4.36, color='gray', linestyle='--', alpha=0.5, label='Mac M3 Max baseline')
    ax.legend()
    ax.set_ylim(0, 7)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig4_throughput_comparison.png')
    print(f'Saved fig4_throughput_comparison.png')
    plt.close()


def fig_activation_heatmap():
    """Fig 2: Expert activation frequency heatmap."""
    profile_path = 'experiments/activation_profile_35B.json'
    if not os.path.exists(profile_path):
        print(f'Skipping heatmap: {profile_path} not found')
        return

    with open(profile_path) as f:
        profile = json.load(f)

    # Build Gini + coverage per layer
    layers = sorted(int(k) for k in profile.keys())
    ginis = [profile[str(l)]['gini'] for l in layers]
    coverages = [profile[str(l)]['top20pct_coverage'] for l in layers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.bar(layers, ginis, color='steelblue', alpha=0.8)
    ax1.set_ylabel('Gini Coefficient')
    ax1.set_title('Expert Activation Inequality by Layer')
    ax1.axhline(y=np.mean(ginis), color='red', linestyle='--', alpha=0.5,
                label=f'Mean: {np.mean(ginis):.3f}')
    ax1.legend()

    ax2.bar(layers, [c*100 for c in coverages], color='darkorange', alpha=0.8)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Top-20% Expert Coverage (%)')
    ax2.set_title('Activation Coverage of Top 20% Experts')
    ax2.axhline(y=np.mean(coverages)*100, color='red', linestyle='--', alpha=0.5,
                label=f'Mean: {np.mean(coverages)*100:.1f}%')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig2_activation_heatmap.png')
    print(f'Saved fig2_activation_heatmap.png')
    plt.close()


def fig_svd_negative_result():
    """Fig 5: SVD energy ratio showing full-rank expert weights."""
    ranks = [8, 16, 32, 64, 128, 256]
    gate_energy = [0.1038, 0.1758, 0.2938, 0.4772, 0.7321, 0.9653]
    up_energy = [0.0933, 0.1635, 0.2814, 0.4671, 0.7266, 0.9644]
    down_energy = [0.0943, 0.1611, 0.2725, 0.4517, 0.7117, 0.9620]

    # Ideal low-rank (for comparison)
    ideal_low_rank = [0.85, 0.95, 0.99, 0.999, 1.0, 1.0]

    fig, ax = plt.subplots()
    ax.plot(ranks, gate_energy, 'o-', label='gate_proj [512,2048]', linewidth=2)
    ax.plot(ranks, up_energy, 's-', label='up_proj [512,2048]', linewidth=2)
    ax.plot(ranks, down_energy, '^-', label='down_proj [2048,512]', linewidth=2)
    ax.plot(ranks, ideal_low_rank, '--', color='gray', alpha=0.5,
            label='Ideal low-rank (eff_rank=20)')

    ax.axhline(y=0.95, color='red', linestyle=':', alpha=0.5)
    ax.text(256, 0.96, 'Target: 95% energy', color='red', ha='right', fontsize=10)

    ax.set_xlabel('SVD Rank')
    ax.set_ylabel('Energy Ratio')
    ax.set_title('SVD Energy Ratio of Real MoE Expert Weights\n(Qwen3.5-35B-A3B, Layer 0, 10 experts)')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.set_xscale('log', base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels(ranks)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig5_svd_negative.png')
    print(f'Saved fig5_svd_negative.png')
    plt.close()


if __name__ == '__main__':
    fig_cache_hit_vs_size()
    fig_throughput_comparison()
    fig_activation_heatmap()
    fig_svd_negative_result()
    print(f'\nAll figures saved to {OUTPUT_DIR}/')
