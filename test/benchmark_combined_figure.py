#!/usr/bin/env python3
"""
Comprehensive benchmarking script to generate data for the combined timing figure.

This script:
1. Benchmarks Full Attention (Dense) baseline
2. Benchmarks TokenButler with multiple K values (e.g., K=1024, K=8192)
3. Measures ALL operations: QKV Proj, Predictor, Scoring, Selection, Key/Value Gather, 
   RoPE, Attention Kernel, MLP, Other
4. Outputs to a unified CSV for plotting
5. Generates the combined academic figure with:
   - Top row: Stacked area charts (Full Attention, TokenButler K=1024, TokenButler K=8192)
   - Bottom left: Scaling dynamics (Baseline Attention vs TokenButler Scoring+Selection vs TokenButler Attention)
   - Bottom right: Speedup plot

Usage:
    python test/benchmark_combined_figure.py --quick  # Fast test run
    python test/benchmark_combined_figure.py          # Full benchmark
    python test/benchmark_combined_figure.py --load-csv test/output/combined_timing.csv --plot-only  # Just plot
"""

import sys
import os
import torch
import gc
import csv
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import Llama
import models.base
from termcolor import colored

# =============================================================================
# Plot Style Configuration (Academic Quality)
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color scheme matching the reference figure
COLORS = {
    'qkv_projection': '#56B4E9',       # Sky Blue - QKV Proj
    'predictor_forward': '#648FFF',    # Blue - Predictor
    'compute_scores': '#785EF0',       # Purple - Scoring
    'topk_selection': '#DC267F',       # Magenta - Selection
    'get_key_cache_total': '#FE6100',  # Orange - Key Gather
    'get_value_cache_total': '#FFB000',# Gold - Value Gather
    'rope_embedding': '#009E73',       # Teal - RoPE
    'flash_attn_compute': '#D55E00',   # Vermilion - Attention Kernel
    'mlp_compute': '#F0E442',          # Yellow - MLP
    'other': '#999999',                # Gray - Other
}

OPERATION_LABELS = {
    'qkv_projection': 'QKV Proj',
    'predictor_forward': 'Predictor',
    'compute_scores': 'Scoring',
    'topk_selection': 'Selection',
    'get_key_cache_total': 'Key Gather',
    'get_value_cache_total': 'Value Gather',
    'rope_embedding': 'RoPE',
    'flash_attn_compute': 'Attention Kernel',
    'mlp_compute': 'MLP',
    'other': 'Other',
}

# =============================================================================
# Profiler
# =============================================================================
class DetailedProfiler:
    """Context manager for detailed CUDA timing with deferred synchronization."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.active = True
        self.pending_events = []
        
    def record(self, name):
        """Returns a context manager for timing a specific operation."""
        class Timer:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
                self.start_event = torch.cuda.Event(enable_timing=True)
                self.end_event = torch.cuda.Event(enable_timing=True)
                
            def __enter__(self):
                if self.profiler.active:
                    self.start_event.record()
                return self
                
            def __exit__(self, *args):
                if self.profiler.active:
                    self.end_event.record()
                    self.profiler.pending_events.append((self.name, self.start_event, self.end_event))
        
        return Timer(self, name)
    
    def step(self):
        """Synchronize and process all pending events for this step."""
        if not self.active or not self.pending_events:
            self.pending_events = []
            return

        torch.cuda.synchronize()
        
        for name, start, end in self.pending_events:
            elapsed = start.elapsed_time(end)  # Already in ms
            self.timings[name].append(elapsed)
            
        self.pending_events = []
    
    def get_stats(self):
        """Get summary statistics for all recorded timings."""
        stats = {}
        for name, times in self.timings.items():
            if times:
                stats[name] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'count': len(times),
                    'sum': np.sum(times),
                }
        return stats


# =============================================================================
# Benchmarking Functions
# =============================================================================
def run_benchmark(prompt_length, gen_length, attn_mode='full', sparse_budget=0, predictor_path=''):
    """
    Run a single benchmark configuration.
    
    Args:
        prompt_length: Context length (prefill tokens)
        gen_length: Number of decode tokens to generate
        attn_mode: 'full' for baseline, 'tokenbutler' for sparse
        sparse_budget: K value for TokenButler (ignored for full)
        predictor_path: Path to TokenButler predictor weights
    
    Returns:
        Dictionary with timing statistics
    """
    mode_str = f"TokenButler K={sparse_budget}" if attn_mode == 'tokenbutler' else "Full Attention"
    print(f"\n  → Context: {prompt_length:,} | {mode_str}")
    
    # Build model kwargs
    model_kwargs = {
        'model_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'batch_size': 1,
        'max_length': 140000,
        'device': 'cuda:0',
        'dtype': torch.bfloat16,
        'attn_mode': attn_mode,
    }
    
    if attn_mode == 'tokenbutler':
        model_kwargs.update({
            'sparse_budget': sparse_budget,
            'chunk_size': 8,
            'rank': 160,
            'dDash': 16,
            'producer_frequency': 4,
            'tokenbutler_intermediate_dim': 512,
            'predictor_path': predictor_path,
        })
    
    llm = Llama(**model_kwargs)
    profiler = DetailedProfiler()
    profiler.active = False  # Don't profile prefill
    
    # Set TokenButler profiler if applicable
    if attn_mode == 'tokenbutler' and hasattr(llm.kv_cache, 'profiler'):
        llm.kv_cache.profiler = profiler
    
    # Monkey-patch for comprehensive profiling
    original_flash_attn = models.base.flash_attn_with_kvcache
    def profiled_flash_attn(*args, **kwargs):
        with profiler.record('flash_attn_compute'):
            return original_flash_attn(*args, **kwargs)
    models.base.flash_attn_with_kvcache = profiled_flash_attn
    
    original_pre_attn = llm.pre_attention_compute
    def profiled_pre_attn(*args, **kwargs):
        with profiler.record('qkv_projection'):
            return original_pre_attn(*args, **kwargs)
    llm.pre_attention_compute = profiled_pre_attn
    
    original_rope = llm.apply_rotary_pos_emb
    def profiled_rope(*args, **kwargs):
        with profiler.record('rope_embedding'):
            return original_rope(*args, **kwargs)
    llm.apply_rotary_pos_emb = profiled_rope
    
    original_post_attn = llm.post_attention_compute
    def profiled_post_attn(*args, **kwargs):
        with profiler.record('mlp_compute'):
            return original_post_attn(*args, **kwargs)
    llm.post_attention_compute = profiled_post_attn
    
    # Create prompt
    base_text = "The quick brown fox jumps over the lazy dog. "
    repetitions = (prompt_length // 10) + 1
    text = base_text * repetitions
    input_ids = llm.encode(text)
    if input_ids.shape[1] > prompt_length:
        input_ids = input_ids[:, :prompt_length]
    
    actual_prompt_len = input_ids.shape[1]
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Prefill
    logits = llm.prefill(input_ids)
    torch.cuda.synchronize()
    llm.kv_cache.H2D()
    torch.cuda.synchronize()
    
    # Decode with profiling
    sample_rate = max(1, gen_length // 100)
    
    for i in range(gen_length):
        torch.cuda.synchronize()
        next_token = logits.argmax(dim=-1)
        profiler.active = (i % sample_rate == 0)
        
        with profiler.record('total_step'):
            position_ids = llm.get_ctx(next_token)
            logits = llm.inference(input_ids=next_token, position_ids=position_ids)
        
        profiler.step()
        torch.cuda.synchronize()
    
    stats = profiler.get_stats()
    
    # Restore
    models.base.flash_attn_with_kvcache = original_flash_attn
    
    del llm
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'context_length': actual_prompt_len,
        'attn_mode': attn_mode,
        'sparse_budget': sparse_budget if attn_mode == 'tokenbutler' else 0,
        'profiling_stats': stats,
    }


def run_all_benchmarks(context_lengths, tokenbutler_k_values, gen_length, predictor_path):
    """
    Run benchmarks for all configurations:
    - Full Attention baseline (all context lengths)
    - TokenButler with each K value (all context lengths)
    """
    all_results = []
    
    # Calculate total configurations
    total_configs = len(context_lengths) + len(context_lengths) * len(tokenbutler_k_values)
    print(f"\n{'='*70}")
    print(f"Running {total_configs} total configurations")
    print(f"  - Full Attention: {len(context_lengths)} contexts")
    print(f"  - TokenButler: {len(context_lengths)} contexts × {len(tokenbutler_k_values)} K values")
    print(f"{'='*70}")
    
    # 1. Full Attention Baseline
    print(f"\n{'='*50}")
    print("Phase 1: Full Attention Baseline")
    print(f"{'='*50}")
    for ctx_len in context_lengths:
        result = run_benchmark(ctx_len, gen_length, attn_mode='full')
        all_results.append(result)
    
    # 2. TokenButler with different K values
    for k in tokenbutler_k_values:
        print(f"\n{'='*50}")
        print(f"Phase 2: TokenButler K={k}")
        print(f"{'='*50}")
        for ctx_len in context_lengths:
            # Skip if K > context/2
            if k > ctx_len // 2:
                print(f"  → Skipping K={k} for context={ctx_len} (K > context/2)")
                continue
            result = run_benchmark(ctx_len, gen_length, attn_mode='tokenbutler', 
                                   sparse_budget=k, predictor_path=predictor_path)
            all_results.append(result)
    
    return all_results


# =============================================================================
# CSV Export
# =============================================================================
def export_to_csv(results, output_path):
    """Export all results to a unified CSV file."""
    
    # All possible operations
    all_operations = [
        'qkv_projection', 'predictor_forward', 'compute_scores', 'topk_selection',
        'get_key_cache_total', 'get_value_cache_total', 'rope_embedding',
        'flash_attn_compute', 'mlp_compute', 'total_step'
    ]
    
    fieldnames = ['context_length', 'attn_mode', 'sparse_budget']
    for op in all_operations:
        fieldnames.extend([f'{op}_mean_ms', f'{op}_std_ms', f'{op}_count'])
    fieldnames.append('total_per_token_ms')
    
    rows = []
    for result in results:
        row = {
            'context_length': result['context_length'],
            'attn_mode': result['attn_mode'],
            'sparse_budget': result['sparse_budget'],
        }
        
        stats = result['profiling_stats']
        total_time = 0
        
        for op in all_operations:
            if op in stats:
                row[f'{op}_mean_ms'] = f"{stats[op]['mean']:.4f}"
                row[f'{op}_std_ms'] = f"{stats[op]['std']:.4f}"
                row[f'{op}_count'] = stats[op]['count']
                if op != 'total_step':
                    total_time += stats[op]['mean']
            else:
                row[f'{op}_mean_ms'] = ''
                row[f'{op}_std_ms'] = ''
                row[f'{op}_count'] = ''
        
        # Use total_step if available, else sum of ops
        if 'total_step' in stats:
            row['total_per_token_ms'] = f"{stats['total_step']['mean']:.4f}"
        else:
            row['total_per_token_ms'] = f"{total_time:.4f}"
        
        rows.append(row)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n✓ Saved CSV: {output_path}")
    return output_path


# =============================================================================
# Plotting Functions
# =============================================================================
def load_csv_for_plotting(csv_path):
    """Load CSV and convert to result-like structure."""
    results = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats = {}
            for key in row:
                if key.endswith('_mean_ms') and row[key]:
                    op_name = key.replace('_mean_ms', '')
                    stats[op_name] = {
                        'mean': float(row[key]),
                        'std': float(row.get(f'{op_name}_std_ms', 0) or 0),
                        'count': int(row.get(f'{op_name}_count', 0) or 0),
                    }
            
            results.append({
                'context_length': int(row['context_length']),
                'attn_mode': row['attn_mode'],
                'sparse_budget': int(row['sparse_budget']) if row['sparse_budget'] else 0,
                'profiling_stats': stats,
            })
    
    return results


def plot_combined_figure(results, output_dir):
    """
    Generate the combined academic figure with:
    - Top row: 3 stacked area charts (Full Attention, TokenButler K=1024, TokenButler K=8192)
    - Bottom row: Scaling dynamics + Speedup plot
    """
    output_dir = Path(output_dir)
    
    # Separate results by mode
    full_results = [r for r in results if r['attn_mode'] == 'full']
    ks_results = [r for r in results if r['attn_mode'] == 'tokenbutler']
    
    # Get unique K values
    k_values = sorted(set(r['sparse_budget'] for r in ks_results if r['sparse_budget'] > 0))
    if not k_values:
        k_values = [1024, 8192]  # Default for display
    
    # Use first two K values for the figure (or all if only 1-2)
    display_k_values = k_values[:2] if len(k_values) >= 2 else k_values
    
    # Operations for stacked area (in stack order, bottom to top)
    ops_tokenbutler = [
        'qkv_projection', 'predictor_forward', 'compute_scores', 'topk_selection',
        'get_key_cache_total', 'get_value_cache_total', 'rope_embedding',
        'flash_attn_compute', 'mlp_compute'
    ]
    ops_full = ['qkv_projection', 'rope_embedding', 'flash_attn_compute', 'mlp_compute']
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.8], hspace=0.35, wspace=0.25)
    
    # =========================================================================
    # Top Row: Stacked Area Charts
    # =========================================================================
    
    def plot_stacked_area(ax, results_subset, operations, title):
        """Plot stacked area chart for a single mode/K configuration."""
        if not results_subset:
            ax.set_title(title + " (No Data)", fontweight='bold')
            return
        
        results_sorted = sorted(results_subset, key=lambda r: r['context_length'])
        ctx_lengths = [r['context_length'] for r in results_sorted]
        
        # Build stacked data
        # IMPORTANT: Operations like qkv_projection, flash_attn run once per layer (32 times per step)
        # So we need: time_per_token = (mean * count) / num_steps
        stacked_data = []
        for op in operations:
            times = []
            for r in results_sorted:
                stats = r['profiling_stats']
                if op in stats:
                    # Get number of decode steps from total_step count
                    num_steps = stats.get('total_step', {}).get('count', 1)
                    if num_steps == 0:
                        num_steps = 1
                    # Total time for this op = mean * count
                    # Time per token = total_time / num_steps
                    op_total = stats[op]['mean'] * stats[op]['count']
                    time_per_token = op_total / num_steps
                    times.append(time_per_token)
                else:
                    times.append(0)
            stacked_data.append(times)
        
        # Calculate "Other" as total - sum of tracked ops
        other_times = []
        for i, r in enumerate(results_sorted):
            stats = r['profiling_stats']
            total = stats.get('total_step', {}).get('mean', 0)
            tracked_sum = sum(stacked_data[j][i] for j in range(len(operations)))
            other_times.append(max(0, total - tracked_sum))
        stacked_data.append(other_times)
        
        stacked_data = np.array(stacked_data)
        
        # Colors and labels
        colors = [COLORS.get(op, '#CCCCCC') for op in operations] + [COLORS['other']]
        labels = [OPERATION_LABELS.get(op, op) for op in operations] + ['Other']
        
        ax.stackplot(ctx_lengths, stacked_data, labels=labels, colors=colors, alpha=0.85)
        ax.set_xlabel('Context Length')
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(min(ctx_lengths), max(ctx_lengths))
        ax.set_ylim(0, None)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    
    # Plot 1: Full Attention
    ax1 = fig.add_subplot(gs[0, 0])
    plot_stacked_area(ax1, full_results, ops_full, 'Full Attention')
    ax1.set_ylabel('Time per Token (ms)')
    
    # Plot 2 & 3: TokenButler with different K values
    for idx, k in enumerate(display_k_values):
        ax = fig.add_subplot(gs[0, idx + 1])
        k_results = [r for r in ks_results if r['sparse_budget'] == k]
        plot_stacked_area(ax, k_results, ops_tokenbutler, f'TokenButler (K={k})')
    
    # If only 1 K value, leave third plot empty or use different K
    if len(display_k_values) < 2:
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.text(0.5, 0.5, 'Only 1 K value\nbenchmarked', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12, color='gray')
        ax3.set_axis_off()
    
    # =========================================================================
    # Bottom Left: Scaling Dynamics
    # =========================================================================
    ax_scaling = fig.add_subplot(gs[1, 0:2])
    
    # Get context lengths
    ctx_lengths = sorted(set(r['context_length'] for r in results))
    
    # Baseline: Full Attention Kernel time
    baseline_attn = []
    for ctx in ctx_lengths:
        r = next((x for x in full_results if x['context_length'] == ctx), None)
        if r and 'flash_attn_compute' in r['profiling_stats']:
            baseline_attn.append(r['profiling_stats']['flash_attn_compute']['mean'])
        else:
            baseline_attn.append(np.nan)
    
    # TokenButler K=1024 (or first K): Scoring + Selection
    first_k = display_k_values[0] if display_k_values else 1024
    ks_scoring_selection = []
    ks_attn = []
    for ctx in ctx_lengths:
        r = next((x for x in ks_results if x['context_length'] == ctx and x['sparse_budget'] == first_k), None)
        if r:
            stats = r['profiling_stats']
            score_sel = (stats.get('compute_scores', {}).get('mean', 0) + 
                        stats.get('topk_selection', {}).get('mean', 0))
            ks_scoring_selection.append(score_sel)
            ks_attn.append(stats.get('flash_attn_compute', {}).get('mean', 0))
        else:
            ks_scoring_selection.append(np.nan)
            ks_attn.append(np.nan)
    
    ax_scaling.plot(ctx_lengths, baseline_attn, 'o-', color='#D55E00', linewidth=2, 
                   markersize=8, label='Baseline: Attention Kernel')
    ax_scaling.plot(ctx_lengths, ks_scoring_selection, 's--', color='#785EF0', linewidth=2,
                   markersize=8, label=f'TokenButler (K={first_k}): Scoring + Selection')
    ax_scaling.plot(ctx_lengths, ks_attn, '^-', color='#009E73', linewidth=2,
                   markersize=8, label=f'TokenButler (K={first_k}): Attention Kernel')
    
    ax_scaling.set_xlabel('Context Length')
    ax_scaling.set_ylabel('Time (ms)')
    ax_scaling.set_title('Scaling Dynamics: Attention vs Scoring', fontweight='bold')
    ax_scaling.legend(loc='upper left', frameon=True)
    ax_scaling.grid(alpha=0.3, linestyle='--')
    ax_scaling.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    
    # =========================================================================
    # Bottom Right: Speedup Plot
    # =========================================================================
    ax_speedup = fig.add_subplot(gs[1, 2])
    
    # Calculate speedup: Baseline Total / TokenButler Total
    speedups = []
    valid_ctx = []
    for ctx in ctx_lengths:
        r_full = next((x for x in full_results if x['context_length'] == ctx), None)
        r_ks = next((x for x in ks_results if x['context_length'] == ctx and x['sparse_budget'] == first_k), None)
        
        if r_full and r_ks:
            full_total = r_full['profiling_stats'].get('total_step', {}).get('mean', 0)
            ks_total = r_ks['profiling_stats'].get('total_step', {}).get('mean', 0)
            if ks_total > 0 and full_total > 0:
                speedups.append(full_total / ks_total)
                valid_ctx.append(ctx)
    
    if speedups:
        ax_speedup.plot(valid_ctx, speedups, 'D-', color='#0072B2', linewidth=2, markersize=8)
        ax_speedup.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline (1.0x)')
        ax_speedup.set_xlabel('Context Length')
        ax_speedup.set_ylabel('Speedup (x)')
        ax_speedup.set_title(f'Speedup (Baseline / TokenButler-{first_k})', fontweight='bold')
        ax_speedup.grid(alpha=0.3, linestyle='--')
        ax_speedup.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
        ax_speedup.legend(loc='upper left')
    else:
        ax_speedup.text(0.5, 0.5, 'Insufficient data\nfor speedup', ha='center', va='center',
                       transform=ax_speedup.transAxes, fontsize=12, color='gray')
    
    # =========================================================================
    # Add Legend for Operations (outside figure)
    # =========================================================================
    all_ops = ops_tokenbutler + ['other']
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=COLORS.get(op, '#CCCCCC'), 
                                      edgecolor='none', alpha=0.85)
                      for op in all_ops]
    legend_labels = [OPERATION_LABELS.get(op, op) for op in all_ops]
    
    # Add legend to the right of the top-right subplot
    fig.legend(legend_elements[::-1], legend_labels[::-1], 
              loc='upper right', bbox_to_anchor=(0.99, 0.98),
              title='Operations', frameon=True, fancybox=True)
    
    # Save
    plt.savefig(output_dir / 'combined_timing_figure.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'combined_timing_figure.pdf', bbox_inches='tight')
    print(f"  → Saved: combined_timing_figure.png/pdf")
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Benchmark and plot combined timing figure for Dense vs Sparse attention',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  python test/benchmark_combined_figure.py --quick --gen-length 64
  
  # Full benchmark
  python test/benchmark_combined_figure.py --gen-length 256
  
  # Just plot from existing CSV
  python test/benchmark_combined_figure.py --load-csv test/output/combined_timing.csv --plot-only
        """
    )
    parser.add_argument('--quick', action='store_true', help='Quick mode with fewer configs')
    parser.add_argument('--gen-length', type=int, default=256, help='Decode tokens per config')
    parser.add_argument('--output-dir', type=str, default='test/output', help='Output directory')
    parser.add_argument('--load-csv', type=str, help='Load existing CSV instead of benchmarking')
    parser.add_argument('--plot-only', action='store_true', help='Only generate plots (requires --load-csv)')
    parser.add_argument('--predictor-path', type=str, default='', help='Path to TokenButler predictor weights')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Define configurations
    if args.quick:
        context_lengths = [4096, 8192, 16384, 32768]
        tokenbutler_k_values = [1024, 4096]
    else:
        context_lengths = [32768, 65536, 131072]
        tokenbutler_k_values = [1024, 4096]
    
    # Run or load
    if args.load_csv and Path(args.load_csv).exists():
        print(f"Loading results from {args.load_csv}")
        results = load_csv_for_plotting(args.load_csv)
    elif args.plot_only:
        print("Error: --plot-only requires --load-csv with a valid file")
        return
    else:
        print(f"\n{'#'*70}")
        print(f"# Combined Benchmark: Dense + TokenButler")
        print(f"# Contexts: {context_lengths}")
        print(f"# K values: {tokenbutler_k_values}")
        print(f"# Gen length: {args.gen_length}")
        print(f"{'#'*70}")
        
        results = run_all_benchmarks(
            context_lengths, 
            tokenbutler_k_values, 
            args.gen_length,
            args.predictor_path
        )
        
        # Export to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f'combined_timing_{timestamp}.csv'
        export_to_csv(results, csv_path)
        
        # Also save as latest
        latest_csv = output_dir / 'combined_timing.csv'
        export_to_csv(results, latest_csv)
    
    # Generate plots
    print(f"\n{'='*70}")
    print("Generating combined figure...")
    print(f"{'='*70}")
    
    plot_combined_figure(results, output_dir)
    
    print(f"\n{colored('✓ All done! Output in ' + str(output_dir), 'green')}")


if __name__ == '__main__':
    main()
