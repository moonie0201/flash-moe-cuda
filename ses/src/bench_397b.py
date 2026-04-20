#!/usr/bin/env python3
"""Benchmark harness for 397B SSD streaming inference variants.

Runs multiple configs, saves structured results to JSON.
Drops OS page cache between cold runs (requires sudo, optional).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = [
    # (label, args)
    ('bf16_cold',    ['--tokens', '5', '--hot-pct', '0']),
    ('bf16_hot30',   ['--tokens', '5', '--hot-pct', '0.3']),
    ('gptq_cold',    ['--tokens', '5', '--hot-pct', '0', '--gptq']),
    ('gptq_hot50',   ['--tokens', '5', '--hot-pct', '0.5', '--gptq', '--cache-format', 'bf16']),
    ('gptq_hot50_raw', ['--tokens', '5', '--hot-pct', '0.5', '--gptq', '--cache-format', 'raw']),
]


def run_one(label, base_args, prompt, out_dir, env=None):
    """Run one config, capture output, parse tok/s."""
    cmd = ['venv/bin/python', 'ses/src/run_397b_ssd.py', '--prompt', prompt] + base_args
    log_path = out_dir / f'{label}.log'
    run_env = os.environ.copy()
    run_env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    if env:
        run_env.update(env)

    print(f'[{label}] {" ".join(cmd)}', flush=True)
    t0 = time.time()
    with open(log_path, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                env=run_env, timeout=2400)
    elapsed = time.time() - t0

    # Parse key metrics from log
    with open(log_path) as f:
        log = f.read()

    def extract(pattern, default=None):
        import re
        m = re.search(pattern, log)
        return m.group(1) if m else default

    cold_toks  = extract(r'\[COLD \(trace\)\] (\d+) tokens in')
    cold_time  = extract(r'\[COLD \(trace\)\] \d+ tokens in ([\d.]+)s')
    cold_tps   = extract(r'\[COLD \(trace\)\] \d+ tokens in [\d.]+s \(([\d.]+) tok/s\)')
    hot_tps    = extract(r'\[HOT\] \d+ tokens in [\d.]+s \(([\d.]+) tok/s\)')
    hot_time   = extract(r'\[HOT\] \d+ tokens in ([\d.]+)s')
    cache_gb   = extract(r'HOT cache: \d+ experts, ~([\d.]+)GB')
    hits       = extract(r'load_hit=[\d.]+s \((\d+)×\)')
    misses     = extract(r'load_miss=[\d.]+s \((\d+)×\)')

    return {
        'label': label,
        'args': base_args,
        'elapsed_total_s': elapsed,
        'exit_code': result.returncode,
        'cold_tokens': int(cold_toks) if cold_toks else None,
        'cold_s': float(cold_time) if cold_time else None,
        'cold_tps': float(cold_tps) if cold_tps else None,
        'hot_s': float(hot_time) if hot_time else None,
        'hot_tps': float(hot_tps) if hot_tps else None,
        'cache_gb': float(cache_gb) if cache_gb else None,
        'hits': int(hits) if hits else None,
        'misses': int(misses) if misses else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='What is MoE? One sentence.')
    parser.add_argument('--out-dir', default='experiments/bench_397b')
    parser.add_argument('--configs', default='all',
                        help='Comma-separated: bf16_cold, gptq_hot50, etc')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.configs == 'all':
        selected = CONFIGS
    else:
        names = set(args.configs.split(','))
        selected = [c for c in CONFIGS if c[0] in names]

    results = []
    for label, base in selected:
        r = run_one(label, base, args.prompt, out_dir)
        results.append(r)
        print(f'[{label}] cold_tps={r["cold_tps"]}, hot_tps={r["hot_tps"]}, '
              f'cache={r["cache_gb"]}GB', flush=True)

    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Markdown table
    md_path = out_dir / 'summary.md'
    with open(md_path, 'w') as f:
        f.write('# 397B Benchmark Summary\n\n')
        f.write(f'Prompt: `{args.prompt}`\n\n')
        f.write('| Config | COLD tok/s | HOT tok/s | Cache GB | Hits | Misses |\n')
        f.write('|--------|-----------|-----------|----------|------|--------|\n')
        for r in results:
            f.write(f'| {r["label"]} | {r["cold_tps"]} | {r["hot_tps"]} | '
                    f'{r["cache_gb"]} | {r["hits"]} | {r["misses"]} |\n')
    print(f'Summary: {summary_path}, {md_path}', flush=True)


if __name__ == '__main__':
    main()
