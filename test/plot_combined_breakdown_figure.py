#!/usr/bin/env python3
"""
Generate a combined academic figure comparing KeySifter and Baseline.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.gridspec as gridspec

# Academic-quality plot settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette
COLORS = {
    'flash_attn_compute': '#D55E00',      # Vermilion (Attn)
    'compute_scores': '#785EF0',          # Purple (Scoring)
    'topk_selection': '#DC267F',          # Magenta (Selection)
    'get_key_cache_total': '#FE6100',     # Orange (Gather)
    'get_value_cache_total': '#FFB000',   # Gold (Gather)
    'qkv_projection': '#56B4E9',          # Sky Blue (Proj)
    'rope_embedding': '#0072B2',          # Blueish Green (RoPE)
    'mlp_compute': '#F0E442',             # Yellow (MLP)
    'prepare_tensors': '#AA4499',         # Purple (Prep)
    'predictor_forward': '#648FFF',       # Blue (Predictor)
    'other_model_ops': '#CCCCCC',         # Grey (Overhead)
}

LABELS = {
    'flash_attn_compute': 'Attention Kernel',
    'compute_scores': 'Scoring',
    'topk_selection': 'Selection',
    'get_key_cache_total': 'Key Gather',
    'get_value_cache_total': 'Value Gather',
    'qkv_projection': 'QKV Proj',
    'rope_embedding': 'RoPE',
    'mlp_compute': 'MLP',
    'predictor_forward': 'Predictor',
    'other_model_ops': 'Other',
}

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_step_count(stats):
    """Get the number of generation steps (tokens) from stats."""
    # Try typical ops that run once per step per layer
    if 'total_step_time' in stats:
        return stats['total_step_time']['count']
    # Fallback: estimate from qkv_projection count / 32 layers
    if 'qkv_projection' in stats:
        return max(1, stats['qkv_projection']['count'] / 32)
    return 1

def get_total_time(stats, operations):
    """Calculate total time per token for a set of operations."""
    total_ms = 0
    # Steps (tokens) is usually 1 if we look at total_step_time which records ONCE per step
    # But other ops record ONCE per LAYER per step.
    # We want time per TOKEN.
    # The 'mean' is time per call.
    # The 'count' is total calls (layers * steps).
    # So total time spent = mean * count.
    # Time per token = (mean * count) / number_of_steps.
    
    steps = get_step_count(stats)
    
    # If total_step_time is available and counts match steps, use it for total
    if 'total_step_time' in stats:
        # total_step_time wraps the whole step, so mean is already time per step
        return stats['total_step_time']['mean'] * 1000
        
    for op in operations:
        if op in stats:
            s = stats[op]
            total_ms += s['mean'] * s['count'] * 1000 
            
    return total_ms / steps

def get_op_time(stats, op):
    """Calculate time per token for a specific operation."""
    if op in stats:
        s = stats[op]
        steps = get_step_count(stats)
        return (s['mean'] * s['count'] * 1000) / steps
    return 0

def plot_stacked_area(ax, results, operations, title, max_y=None):
    """Plot a single stacked area chart."""
    # Filter and sort by context
    sorted_res = sorted(results, key=lambda x: x['prompt_length'])
    ctx = [r['prompt_length'] for r in sorted_res]
    
    stacked_data = []
    
    # Pre-calculate totals per run to handle 'other' calculation
    run_totals = {}
    for r in sorted_res:
         if 'total_step_time' in r['profiling_stats']:
             # Direct total measurement
             steps = get_step_count(r['profiling_stats'])
             s = r['profiling_stats']['total_step_time']
             run_totals[id(r)] = (s['mean'] * s['count'] * 1000) / steps
         else:
             # Fallback sum if total not recorded (shouldn't happen with new scripts)
             run_totals[id(r)] = 0
    
    for op in operations:
        times = []
        for r in sorted_res:
            if op == 'other_model_ops':
                # Calculate Other = Total - Sum(Everything Else)
                total_ms = run_totals[id(r)]
                
                # Sum trackable components in the *requested operations list*
                # Note: We must sum EVERYTHING tracked in stats that corresponds to a component,
                # but essentially we just sum the other ops in the list we are stacking.
                # Ideally we sum everything in stats except total, but let's stick to the ops list for consistency with original.
                
                tracked_sum = 0
                for other_op in operations:
                    if other_op != 'other_model_ops':
                        tracked_sum += get_op_time(r['profiling_stats'], other_op)
                
                other_ms = max(0, total_ms - tracked_sum)
                times.append(other_ms)
            else:
                times.append(get_op_time(r['profiling_stats'], op))
        stacked_data.append(times)
    
    stacked_data = np.array(stacked_data)
    
    # Plot
    colors = [COLORS.get(op, '#DDDDDD') for op in operations]
    labels = [LABELS.get(op, op) for op in operations]
    
    ax.stackplot(ctx, stacked_data, labels=labels, colors=colors, alpha=0.9)
    
    ax.set_title(title, fontweight='bold')
    if max_y:
        ax.set_ylim(0, max_y)
    ax.set_xlim(min(ctx), max(ctx))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    
    return stacked_data.sum(axis=0).max()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-json', required=True)
    parser.add_argument('--keysifter-json', required=True)
    parser.add_argument('--output', default='test/output/combined_figure.png')
    args = parser.parse_args()
    
    # Load data
    full_data = load_data(args.full_json)
    ks_data = load_data(args.keysifter_json)
    
    # Filter for specific topK (1024 and 8192)
    ks_1024 = [r for r in ks_data if r['sparse_budget'] == 1024]
    ks_8192 = [r for r in ks_data if r['sparse_budget'] == 8192]
    
    # Common operations for stacking
    ops_full = ['qkv_projection', 'rope_embedding', 'flash_attn_compute', 'mlp_compute', 'other_model_ops']
    ops_ks = ['qkv_projection', 'predictor_forward', 'compute_scores', 'topk_selection', 
              'get_key_cache_total', 'get_value_cache_total', 'rope_embedding', 
              'flash_attn_compute', 'mlp_compute', 'other_model_ops']
    
    # Setup Figure
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1], hspace=0.35, wspace=0.3)
    
    # --- Top Row: Breakdown ---
    ax_full = fig.add_subplot(gs[0, 0])
    ax_ks1 = fig.add_subplot(gs[0, 1])
    ax_ks2 = fig.add_subplot(gs[0, 2])
    
    # Determine common Y max for comparison
    # Calculate max possible time across all
    max_t1 = plot_stacked_area(ax_full, full_data, ops_full, "Full Attention", max_y=None)
    max_t2 = plot_stacked_area(ax_ks1, ks_1024, ops_ks, "KeySifter (K=1024)", max_y=None)
    max_t3 = plot_stacked_area(ax_ks2, ks_8192, ops_ks, "KeySifter (K=8192)", max_y=None)
    
    global_max = max(max_t1, max_t2, max_t3) * 1.1
    ax_full.set_ylim(0, global_max)
    ax_ks1.set_ylim(0, global_max)
    ax_ks2.set_ylim(0, global_max)
    
    # Hide Y labels for inner plots
    ax_ks1.set_yticklabels([])
    ax_ks2.set_yticklabels([])
    
    ax_full.set_ylabel("Time per Token (ms)")
    ax_ks1.set_xlabel("Context Length")
    
    # Legend for Top Row (in 4th column space)
    ax_leg = fig.add_subplot(gs[0, 3])
    ax_leg.axis('off')
    
    # Collect handles from KS plot (superset of ops)
    h, l = ax_ks1.get_legend_handles_labels()
    ax_leg.legend(h[::-1], l[::-1], loc='center left', title="Operations")
    
    
    # --- Bottom Row: Scaling Analysis ---
    ax_scale = fig.add_subplot(gs[1, :2])
    
    # Plot 1: Full Attn Kernel growth
    full_sorted = sorted(full_data, key=lambda x: x['prompt_length'])
    ctx_full = [r['prompt_length'] for r in full_sorted]
    attn_full = [get_op_time(r['profiling_stats'], 'flash_attn_compute') for r in full_sorted]
    
    ax_scale.plot(ctx_full, attn_full, 'o-', color=COLORS['flash_attn_compute'], 
                 linewidth=2.5, label='Baseline: Attention Kernel')
    
    # Plot 2: KS Score Score + Selection (The "Cost")
    ks_sorted_1024 = sorted(ks_1024, key=lambda x: x['prompt_length'])
    ctx_ks = [r['prompt_length'] for r in ks_sorted_1024]
    
    # Cost = Score + Selection + Gather overhead
    cost_1024 = [get_op_time(r['profiling_stats'], 'compute_scores') + 
                 get_op_time(r['profiling_stats'], 'topk_selection') 
                 for r in ks_sorted_1024]
                 
    ax_scale.plot(ctx_ks, cost_1024, 's--', color=COLORS['compute_scores'], 
                 linewidth=2, label='KeySifter (K=1024): Scoring + Selection')
                 
    # Plot 3: KS Attn (The "Benefit" - should be constant)
    attn_1024 = [get_op_time(r['profiling_stats'], 'flash_attn_compute') for r in ks_sorted_1024]
    ax_scale.plot(ctx_ks, attn_1024, '^-', color='#44AA99', 
                 linewidth=2, label='KeySifter (K=1024): Attention Kernel')

    ax_scale.set_title("Scaling Dynamics: Attention vs Scoring", fontweight='bold')
    ax_scale.set_xlabel("Context Length")
    ax_scale.set_ylabel("Time (ms)")
    ax_scale.grid(True, alpha=0.3)
    ax_scale.legend()
    ax_scale.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    
    
    # --- Bottom Row: Speedup ---
    ax_speed = fig.add_subplot(gs[1, 2:])
    
    # Map context to total time for lookup
    full_map = {r['prompt_length']: get_total_time(r['profiling_stats'], ops_full) for r in full_sorted}
    
    speedups = []
    contexts = []
    
    for r in ks_sorted_1024:
        ctx = r['prompt_length']
        if ctx in full_map:
            t_full = full_map[ctx]
            t_ks = get_total_time(r['profiling_stats'], ops_ks)
            if t_ks > 0:
                speedups.append(t_full / t_ks)
                contexts.append(ctx)
                
    ax_speed.plot(contexts, speedups, 'D-', color='#332288', linewidth=2.5)
    ax_speed.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    
    ax_speed.set_title("Speedup (Baseline / KeySifter-1024)", fontweight='bold')
    ax_speed.set_xlabel("Context Length")
    ax_speed.set_ylabel("Speedup (x)")
    ax_speed.grid(True, alpha=0.3)
    ax_speed.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    
    
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
