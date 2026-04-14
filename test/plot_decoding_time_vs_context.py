#!/usr/bin/env python3
"""
Plot Decoding Time vs Context Length for different configurations:
Dense, KeySifter, KeySifter (i=8+neighbor), DSA, Oracle.
"""

import time
import matplotlib.pyplot as plt
import sys
import os
import argparse
import json
import csv
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Add test directory to sys.path to allow importing sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_keysifter import benchmark_model

from termcolor import colored

def run_benchmarks(args, csv_file):
    """Run benchmarks for all configurations and context lengths.

    Results are written incrementally to csv_file as each experiment completes.
    """

    # Configurations
    context_lengths = [32768, 65536, 131072, 262144, 524288, 1048576] # 32K, 64K, 128K, 256K, 512K, 1M
    if args.quick:
        context_lengths = [2048, 4096]

    # Threshold for switching to CPU offloading (128K tokens)
    CPU_OFFLOAD_THRESHOLD = 128 * 1024

    configs = [
        {'label': 'Dense', 'mode': 'full', 'mode_cpu': 'full_cpu', 'kwargs': {}},
        {'label': 'KeySifter (i=1)', 'mode': 'keysifter', 'mode_cpu': 'keysifter_cpu', 'kwargs': {'sparse_budget': 8192, 'predictor_path': args.predictor_path}},
        {'label': 'KeySifter (i=2+nb)', 'mode': 'keysifter', 'mode_cpu': 'keysifter_cpu', 'kwargs': {'sparse_budget': 8192, 'predictor_path': args.predictor_path, 'predict_interval': 2, 'enable_neighbor_fetch': True}},
        {'label': 'KeySifter (i=4+nb)', 'mode': 'keysifter', 'mode_cpu': 'keysifter_cpu', 'kwargs': {'sparse_budget': 8192, 'predictor_path': args.predictor_path, 'predict_interval': 4, 'enable_neighbor_fetch': True}},
        {'label': 'KeySifter (i=8+nb)', 'mode': 'keysifter', 'mode_cpu': 'keysifter_cpu', 'kwargs': {'sparse_budget': 8192, 'predictor_path': args.predictor_path, 'predict_interval': 8, 'enable_neighbor_fetch': True}},
        {'label': 'KeySifter (i=16+nb)', 'mode': 'keysifter', 'mode_cpu': 'keysifter_cpu', 'kwargs': {'sparse_budget': 8192, 'predictor_path': args.predictor_path, 'predict_interval': 16, 'enable_neighbor_fetch': True}},
        {'label': 'Oracle', 'mode': 'oracle', 'mode_cpu': 'oracle_cpu', 'kwargs': {'sparse_budget': 8192, 'oracle_random_indices': True}},
    ]

    results = {cfg['label']: {'x': [], 'y': []} for cfg in configs}

    # Load existing CSV results for resume support
    completed = set()
    if os.path.exists(csv_file):
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status') == 'success':
                    key = (row['label'], int(row['context_length']))
                    completed.add(key)
                    label = row['label']
                    if label in results:
                        results[label]['x'].append(int(row['context_length']))
                        results[label]['y'].append(float(row['avg_decode_time_ms']))
        if completed:
            print(f"{colored(f'Resuming: {len(completed)} results loaded from {csv_file}', 'cyan')}")
    else:
        # Initialize CSV file with header
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['label', 'context_length', 'mode', 'avg_decode_time_ms', 'status'])
    print(f"{colored(f'Results will be written to: {csv_file}', 'green')}")

    for ctx_len in context_lengths:
        print(f"\n{colored(f'>>> Benchmarking Context Length: {ctx_len}', 'yellow', attrs=['bold'])}")

        # Determine if we should use CPU offloading for this context length
        use_cpu_offload = ctx_len > CPU_OFFLOAD_THRESHOLD
        if use_cpu_offload:
            print(f"  {colored('(Using CPU offloading for contexts > 128K)', 'cyan')}")

        for cfg in configs:
            label = cfg['label']
            # Skip already-completed configurations (resume support)
            if (label, ctx_len) in completed:
                print(f"  Skipping {label} at {ctx_len} (already completed)")
                continue
            # Skip Dense at contexts > 262144 (OOM-kills the process on 48GB GPU)
            if label == 'Dense' and ctx_len > 262144:
                print(f"  Skipping {label} at {ctx_len} (would OOM on GPU)")
                continue
            # Select appropriate mode based on context length
            mode = cfg['mode_cpu'] if use_cpu_offload else cfg['mode']
            print(f"  Running {label} (mode: {mode})...")

            try:
                kwargs = cfg['kwargs'].copy()

                # Extract specific args if they exist in kwargs, else defaults
                sb = kwargs.get('sparse_budget', 2048)
                pp = kwargs.get('predictor_path', '')
                ori = kwargs.get('oracle_random_indices', True)
                pi = kwargs.get('predict_interval', 1)
                enf = kwargs.get('enable_neighbor_fetch', False)

                # Use fewer decode steps for CPU-offloaded contexts (very slow)
                gen_len = 128 if use_cpu_offload else args.gen_length

                res = benchmark_model(
                    attn_mode=mode,
                    prompt_length=ctx_len,
                    gen_length=gen_len,
                    sparse_budget=sb,
                    predictor_path=pp,
                    oracle_random_indices=ori,
                    predict_interval=pi,
                    enable_neighbor_fetch=enf,
                )

                # Collect result
                avg_time_ms = res['decode_time_avg'] * 1000.0
                results[label]['x'].append(ctx_len)
                results[label]['y'].append(avg_time_ms)

                # Write result to CSV immediately
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([label, ctx_len, mode, f'{avg_time_ms:.4f}', 'success'])
                print(f"    {colored(f'Result: {avg_time_ms:.4f} ms (written to CSV)', 'green')}")

            except Exception as e:
                print(f"{colored(f'Failed {label} at {ctx_len}: {e}', 'red')}")
                # Write failed result to CSV
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([label, ctx_len, mode, '', f'failed: {e}'])

    return results

def plot_results(results, output_file):
    """Plot the results."""
    plt.figure(figsize=(12, 7))

    # Styling
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Avg Decoding Time per Token (ms)', fontsize=12)
    plt.title('Decoding Time vs Context Length: KeySifter vs DSA', fontsize=14, fontweight='bold')

    markers = ['o', 's', '^', 'D', 'v', 'p']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linestyles = ['-', '-', '--', '-', '--', ':']

    for idx, (label, data) in enumerate(results.items()):
        if not data['x']:
            continue
        plt.plot(data['x'], data['y'],
                 marker=markers[idx % len(markers)],
                 color=colors[idx % len(colors)],
                 linestyle=linestyles[idx % len(linestyles)],
                 linewidth=2, markersize=8, label=label)

    plt.legend(fontsize=10, loc='upper left')
    plt.xscale('log', base=2)

    # Set nice X ticks if possible
    all_x = sorted(list(set([x for r in results.values() for x in r['x']])))
    if all_x:
        plt.xticks(all_x, [f'{x//1024}K' for x in all_x])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n{colored(f'Plot saved to {output_file}', 'green')}")


def main():
    parser = argparse.ArgumentParser(description='Plot Decoding Time vs Context Length: KeySifter vs DSA')
    parser.add_argument('--quick', action='store_true', help='Run a quick test with fewer/smaller contexts')
    parser.add_argument('--gen-length', type=int, default=1024, help='Number of tokens to generate for averaging')
    parser.add_argument('--predictor-path', type=str, default='', help='Path to KeySifter weights (empty for random weights)')
    parser.add_argument('--output', type=str, default='decoding_time_vs_context.png', help='Output image file')
    parser.add_argument('--csv', type=str, default=f'decoding_time_vs_context_{int(time.time())}.csv', help='Output CSV file for incremental results')
    parser.add_argument('--plot-only', action='store_true', help='Only generate plot from existing CSV (no benchmarking)')

    args = parser.parse_args()

    if args.plot_only:
        # Load results from CSV and plot without running benchmarks
        results = {}
        with open(args.csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status') != 'success':
                    continue
                label = row['label']
                if label not in results:
                    results[label] = {'x': [], 'y': []}
                results[label]['x'].append(int(row['context_length']))
                results[label]['y'].append(float(row['avg_decode_time_ms']))
        total_loaded = sum(len(v['x']) for v in results.values())
        print(f"{colored(f'Loaded {total_loaded} results from {args.csv}', 'green')}")
        plot_results(results, args.output)
        return

    # Check if weights exist (only warn if path was explicitly provided)
    if args.predictor_path and not os.path.exists(args.predictor_path):
        print(f"{colored(f'Warning: Predictor weights not found at {args.predictor_path}', 'red')}")
    elif not args.predictor_path:
        print(f"{colored('Using random KeySifter predictor weights (no checkpoint loaded)', 'cyan')}")

    results = run_benchmarks(args, args.csv)
    plot_results(results, args.output)

if __name__ == '__main__':
    main()
