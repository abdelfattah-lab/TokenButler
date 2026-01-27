#!/usr/bin/env python3
"""
Plot Decoding Time vs Context Length for different configurations:
Dense, KeySifter, Oracle (Random), Oracle (Contiguous).
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
    context_lengths = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
    # context_lengths = [ 1048576 ]  # 256K, 512K, 1M
    if args.quick:
        context_lengths = [2048, 4096]
    
    # Threshold for switching to CPU offloading (128K tokens)
    CPU_OFFLOAD_THRESHOLD = 128 * 1024
        
    configs = [
        {'label': 'KeySifter', 'mode': 'keysifter', 'mode_cpu': 'keysifter_cpu', 'kwargs': {'sparse_budget': 2048, 'predictor_path': args.predictor_path}},
        # {'label': 'Dense', 'mode': 'full', 'mode_cpu': 'full_cpu', 'kwargs': {}},
        # {'label': 'Oracle (Random)', 'mode': 'oracle', 'mode_cpu': 'oracle_cpu', 'kwargs': {'sparse_budget': 2048, 'oracle_random_indices': True}},
        # {'label': 'Oracle (Contiguous)', 'mode': 'oracle', 'mode_cpu': 'oracle_cpu', 'kwargs': {'sparse_budget': 2048, 'oracle_random_indices': False}},
    ]
    
    results = {cfg['label']: {'x': [], 'y': []} for cfg in configs}
    
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
            # Select appropriate mode based on context length
            mode = cfg['mode_cpu'] if use_cpu_offload else cfg['mode']
            print(f"  Running {label} (mode: {mode})...")
            
            # Skip if sparse_budget > context length / 2? 
            # benchmark_keysifter doesn't explicitly check this but it's good practice.
            # However, for consistency, we'll try to run it. 
            # If context is too small for sparse budget, it might just be full attention or error out?
            # KeySifter usually handles it, but let's just run it.
            
            try:
                # Run benchmark
                # benchmark_model(attn_mode, prompt_length, gen_length, sparse_budget=512, predictor_path='', oracle_random_indices=True)
                # We need to unpack kwargs and map to function args
                
                kwargs = cfg['kwargs'].copy()
                # mode is already set above based on context length
                
                # Extract specific args if they exist in kwargs, else defaults
                sb = kwargs.get('sparse_budget', 2048)
                pp = kwargs.get('predictor_path', '')
                ori = kwargs.get('oracle_random_indices', True)
                
                res = benchmark_model(
                    attn_mode=mode,
                    prompt_length=ctx_len,
                    gen_length=args.gen_length,
                    sparse_budget=sb,
                    predictor_path=pp,
                    oracle_random_indices=ori
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
    plt.figure(figsize=(10, 6))
    
    # Styling
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Avg Decoding Time per Token (ms)', fontsize=12)
    plt.title('Decoding Time vs Context Length', fontsize=14, fontweight='bold')
    
    markers = ['o', 's', '^', 'D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Blue, Orange, Green, Red
    
    for idx, (label, data) in enumerate(results.items()):
        if not data['x']:
            continue
        plt.plot(data['x'], data['y'], marker=markers[idx % len(markers)], 
                 color=colors[idx % len(colors)], linewidth=2, label=label)
                 
    plt.legend(fontsize=10)
    plt.xscale('log', base=2) # Often linear or log. Let's start with linear but log base 2 ticks might be nice
    
    # Set nice X ticks if possible
    all_x = sorted(list(set([x for r in results.values() for x in r['x']])))
    if all_x:
        plt.xticks(all_x, [f'{x}' for x in all_x])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\n{colored(f'Plot saved to {output_file}', 'green')}")


def main():
    parser = argparse.ArgumentParser(description='Plot Decoding Time vs Context Length')
    parser.add_argument('--quick', action='store_true', help='Run a quick test with fewer/smaller contexts')
    parser.add_argument('--gen-length', type=int, default=1024, help='NumberOf tokens to generate for averaging')
    parser.add_argument('--predictor-path', type=str, default='/home/afa55/Projects/xKV/xKV/Llama_31_8bi_GQA_dDash16.pt', help='Path to KeySifter weights')
    parser.add_argument('--output', type=str, default='decoding_time_vs_context.png', help='Output image file')
    parser.add_argument('--csv', type=str, default=f'decoding_time_vs_context_{int(time.time())}.csv', help='Output CSV file for incremental results')
    
    args = parser.parse_args()
    
    # Check if weights exist
    if not os.path.exists(args.predictor_path):
        print(f"{colored(f'Warning: Predictor weights not found at {args.predictor_path}', 'red')}")
    
    results = run_benchmarks(args, args.csv)
    plot_results(results, args.output)

if __name__ == '__main__':
    main()
