#!/usr/bin/env python3
"""
Visualize KeySifter timing breakdown trends across context lengths and topK values.

Creates academic-quality visualizations showing how different operations
(predictor, scoring, topK selection, gathering) scale with parameters.
"""

import torch
import time
import sys
import gc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import argparse
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, '/home/afa55/Projects/xKV/xKV')

from models import Llama
from termcolor import colored

# Academic-quality plot settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colorblind-friendly palette (IBM Design Library)
COLORS = {
    'predictor_forward': '#648FFF',      # Blue
    'compute_scores': '#785EF0',          # Purple  
    'topk_selection': '#DC267F',          # Magenta
    'get_key_cache_total': '#FE6100',     # Orange
    'get_value_cache_total': '#FFB000',   # Gold
    'get_retrieval_total': '#009E73',     # Teal
    'update_kv_cache_total': '#999999',   # Gray
    'other': '#CCCCCC',                   # Light gray
}

OPERATION_LABELS = {
    'predictor_forward': 'Predictor Forward',
    'compute_scores': 'Score Computation',
    'topk_selection': 'TopK Selection',
    'get_key_cache_total': 'Key Gathering',
    'get_value_cache_total': 'Value Gathering',
    'get_retrieval_total': 'Retrieval Total',
    'update_kv_cache_total': 'KV Update',
}


class DetailedProfiler:
    """Context manager for detailed CUDA timing."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.active = True
        
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
                    torch.cuda.synchronize()
                    elapsed = self.start_event.elapsed_time(self.end_event) / 1000.0
                    self.profiler.timings[self.name].append(elapsed)
        
        return Timer(self, name)
    
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
                }
        return stats


def monkey_patch_keysifter_for_profiling(llm, profiler):
    """Add profiling instrumentation to KeySifter operations."""
    
    if not hasattr(llm.kv_cache, 'compute_predictor_importance'):
        print("Warning: Not a KeySifter cache, skipping profiling")
        return
    
    original_compute_predictor = llm.kv_cache.compute_predictor_importance
    original_get_retrieval = llm.kv_cache.get_retrieval_position_ids
    original_get_key = llm.kv_cache.get_key_cache
    original_get_value = llm.kv_cache.get_value_cache
    original_update_kv = llm.kv_cache.update_kv_cache
    
    def profiled_compute_predictor(hidden_states, producer_layer_idx):
        with profiler.record('predictor_forward'):
            return original_compute_predictor(hidden_states, producer_layer_idx)
    
    def profiled_get_retrieval(layer_idx, query_states):
        with profiler.record('get_retrieval_total'):
            bsz = query_states.shape[0]
            slot_idx = layer_idx % llm.kv_cache.producer_frequency
            
            with profiler.record('get_importance_query'):
                q_slot = llm.kv_cache.q_importance_cache[:, slot_idx, :, :]
                Lq = q_slot.shape[1]
                q_slot = q_slot.view(bsz, llm.kv_cache.num_attention_heads, Lq, llm.kv_cache.dDash)
                q_slot = q_slot[:, :, -1:, :]
                q_slot = q_slot.view(bsz, llm.kv_cache.num_key_value_heads, llm.kv_cache.num_key_value_groups, 1, llm.kv_cache.dDash)

            with profiler.record('get_projected_keys'):
                k_proj = llm.kv_cache.k_proj_cache[layer_idx, :, :, :llm.kv_cache.last_projected_pos]
                k_proj = k_proj.unsqueeze(2)
            
            with profiler.record('compute_scores'):
                scores = torch.einsum("bhgqd,bhgkd->bhgqk", q_slot, k_proj)
                scores = scores.squeeze(3) / np.sqrt(llm.kv_cache.dDash)
                scores = scores.max(dim=2).values
            
            with profiler.record('mask_local_window'):
                local_start = max(0, llm.kv_cache.kv_offset - llm.kv_cache.local_window)
                if local_start < llm.kv_cache.last_projected_pos:
                    scores[:, :, local_start:] = float("-inf")
            
            with profiler.record('topk_selection'):
                # Original TopK
                num_available = min(local_start, llm.kv_cache.last_projected_pos)
                num_to_select = min(llm.kv_cache.sparse_budget, num_available)
                if num_to_select > 0:
                    _, position_ids = torch.topk(scores, k=num_to_select, dim=-1)
                    position_ids, _ = position_ids.sort(dim=-1)
                else:
                    position_ids = torch.zeros(
                        bsz, llm.kv_cache.num_key_value_heads, 0,
                        device=llm.kv_cache.device, dtype=torch.long
                    )

                # # Random TopK
                # num_available = min(local_start, llm.kv_cache.last_projected_pos)
                # num_to_select = min(llm.kv_cache.sparse_budget, num_available)

                # if num_to_select > 0:
                #     # Generate random positions from [0, num_available) and sort them
                #     # Use same positions for all batches and KV heads for simplicity
                #     random_positions = torch.randperm(num_available, device=llm.kv_cache.device)[:num_to_select]
                #     random_positions, _ = random_positions.sort()

                #     # Expand to [bsz, num_kv_heads, num_to_select]
                #     position_ids = random_positions.unsqueeze(0).unsqueeze(0).expand(bsz, llm.kv_cache.num_key_value_heads, -1)
                # else:
                #     position_ids = torch.zeros(
                #         bsz, llm.kv_cache.num_key_value_heads, 0,
                #         device=llm.kv_cache.device, dtype=torch.long
                #     )
            
            return position_ids
    
    def profiled_get_key(layer_idx, position_ids, rope_func=None, cos_sin_cache=None):
        with profiler.record('get_key_cache_total'):
            return original_get_key(layer_idx, position_ids, rope_func, cos_sin_cache)
    
    def profiled_get_value(layer_idx, position_ids):
        with profiler.record('get_value_cache_total'):
            return original_get_value(layer_idx, position_ids)
    
    def profiled_update_kv(new_k_cache, new_v_cache, layer_idx):
        with profiler.record('update_kv_cache_total'):
            return original_update_kv(new_k_cache, new_v_cache, layer_idx)
    
    llm.kv_cache.compute_predictor_importance = profiled_compute_predictor
    llm.kv_cache.get_retrieval_position_ids = profiled_get_retrieval
    llm.kv_cache.get_key_cache = profiled_get_key
    llm.kv_cache.get_value_cache = profiled_get_value
    llm.kv_cache.update_kv_cache = profiled_update_kv


def run_single_benchmark(prompt_length, gen_length, sparse_budget, predictor_path):
    """Run benchmark for a single configuration."""
    print(f"\n  → Context: {prompt_length:,} | topK: {sparse_budget:,}")
    
    model_kwargs = {
        'model_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'batch_size': 1,
        'max_length': 136072,
        'device': 'cuda:0',
        'dtype': torch.bfloat16,
        'attn_mode': 'keysifter',
        'sparse_budget': sparse_budget,
        'chunk_size': 8,
        'rank': 160,
        'dDash': 8,
        'producer_frequency': 4,
        'keysifter_intermediate_dim': 1024,
        'predictor_path': predictor_path,
    }

    llm = Llama(**model_kwargs)
    profiler = DetailedProfiler()
    monkey_patch_keysifter_for_profiling(llm, profiler)

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
    sample_rate = max(1, gen_length // 50)
    
    for i in range(gen_length):
        torch.cuda.synchronize()
        next_token = logits.argmax(dim=-1)
        profiler.active = (i % sample_rate == 0)
        position_ids = llm.get_ctx(next_token)
        logits = llm.inference(input_ids=next_token, position_ids=position_ids)
        torch.cuda.synchronize()

    stats = profiler.get_stats()

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'prompt_length': actual_prompt_len,
        'sparse_budget': sparse_budget,
        'profiling_stats': stats,
    }


def collect_sweep_data(context_lengths, topk_values, gen_length, predictor_path):
    """Run benchmarks across all configurations."""
    results = []
    
    total_configs = len(context_lengths) * len(topk_values)
    print(f"\n{'='*60}")
    print(f"Running {total_configs} configurations...")
    print(f"{'='*60}")
    
    for ctx_len in context_lengths:
        for topk in topk_values:
            if topk > ctx_len // 2:  # Skip invalid configs
                print(f"  → Skipping topK={topk} > context/2 for ctx={ctx_len}")
                continue
            
            result = run_single_benchmark(ctx_len, gen_length, topk, predictor_path)
            results.append(result)
    
    return results


def extract_timing_matrix(results, operations):
    """Extract timing data into matrices for visualization."""
    # Get unique context lengths and topK values
    ctx_lengths = sorted(set(r['prompt_length'] for r in results))
    topk_values = sorted(set(r['sparse_budget'] for r in results))
    
    # Create matrices: rows = operations, cols = (ctx, topk) pairs
    configs = []
    for ctx in ctx_lengths:
        for topk in topk_values:
            matching = [r for r in results 
                       if r['prompt_length'] == ctx and r['sparse_budget'] == topk]
            if matching:
                configs.append((ctx, topk, matching[0]))
    
    n_ops = len(operations)
    n_configs = len(configs)
    
    time_matrix = np.zeros((n_ops, n_configs))
    pct_matrix = np.zeros((n_ops, n_configs))
    
    for j, (ctx, topk, result) in enumerate(configs):
        stats = result['profiling_stats']
        # Calculate total time as sum of (mean * count) for each operation
        total_time = sum(stats[op]['mean'] * stats[op]['count'] for op in operations if op in stats)
        
        for i, op in enumerate(operations):
            if op in stats:
                # Use total accumulated time (mean * count)
                time_ms = stats[op]['mean'] * stats[op]['count'] * 1000
                time_matrix[i, j] = time_ms
                pct_matrix[i, j] = (time_ms / (total_time * 1000)) * 100 if total_time > 0 else 0
    
    return time_matrix, pct_matrix, configs, ctx_lengths, topk_values


def plot_stacked_area_by_context(results, operations, output_dir):
    """Create stacked area chart: time breakdown vs context length, faceted by topK."""
    topk_values = sorted(set(r['sparse_budget'] for r in results))
    
    n_topk = len(topk_values)
    fig, axes = plt.subplots(1, n_topk, figsize=(4.5 * n_topk, 5), sharey=True)
    if n_topk == 1:
        axes = [axes]
    
    for ax_idx, topk in enumerate(topk_values):
        ax = axes[ax_idx]
        
        # Filter results for this topK
        filtered = [r for r in results if r['sparse_budget'] == topk]
        filtered = sorted(filtered, key=lambda r: r['prompt_length'])
        
        if not filtered:
            continue
        
        ctx_lengths = [r['prompt_length'] for r in filtered]
        
        # Build stacked data
        stacked_data = []
        for op in operations:
            times = []
            for r in filtered:
                if op in r['profiling_stats']:
                    s = r['profiling_stats'][op]
                    times.append(s['mean'] * s['count'] * 1000)
                else:
                    times.append(0)
            stacked_data.append(times)
        
        stacked_data = np.array(stacked_data)
        
        # Compute percentages
        totals = stacked_data.sum(axis=0)
        pct_data = (stacked_data / totals[np.newaxis, :]) * 100
        
        # Plot stacked area
        colors = [COLORS.get(op, COLORS['other']) for op in operations]
        ax.stackplot(ctx_lengths, pct_data, labels=[OPERATION_LABELS.get(op, op) for op in operations],
                    colors=colors, alpha=0.85)
        
        ax.set_xlabel('Context Length (tokens)')
        ax.set_title(f'topK = {topk:,}', fontweight='bold')
        ax.set_xlim(min(ctx_lengths), max(ctx_lengths))
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Format x-axis with K suffix
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    
    axes[0].set_ylabel('Time Breakdown (%)')
    
    # Legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='center right', bbox_to_anchor=(1.15, 0.5),
              frameon=True, fancybox=True, shadow=True)
    
    fig.suptitle('KeySifter Operation Time Breakdown vs Context Length', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'timing_stacked_by_context.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'timing_stacked_by_context.pdf', bbox_inches='tight')
    print(f"  → Saved: timing_stacked_by_context.png/pdf")
    plt.close()


def plot_stacked_area_by_topk(results, operations, output_dir):
    """Create stacked area chart: time breakdown vs topK, faceted by context length."""
    ctx_lengths = sorted(set(r['prompt_length'] for r in results))
    
    n_ctx = len(ctx_lengths)
    fig, axes = plt.subplots(1, n_ctx, figsize=(4.5 * n_ctx, 5), sharey=True)
    if n_ctx == 1:
        axes = [axes]
    
    for ax_idx, ctx in enumerate(ctx_lengths):
        ax = axes[ax_idx]
        
        filtered = [r for r in results if r['prompt_length'] == ctx]
        filtered = sorted(filtered, key=lambda r: r['sparse_budget'])
        
        if not filtered:
            continue
        
        topk_values = [r['sparse_budget'] for r in filtered]
        
        stacked_data = []
        for op in operations:
            times = []
            for r in filtered:
                if op in r['profiling_stats']:
                    s = r['profiling_stats'][op]
                    times.append(s['mean'] * s['count'] * 1000)
                else:
                    times.append(0)
            stacked_data.append(times)
        
        stacked_data = np.array(stacked_data)
        totals = stacked_data.sum(axis=0)
        pct_data = (stacked_data / totals[np.newaxis, :]) * 100
        
        colors = [COLORS.get(op, COLORS['other']) for op in operations]
        ax.stackplot(topk_values, pct_data, labels=[OPERATION_LABELS.get(op, op) for op in operations],
                    colors=colors, alpha=0.85)
        
        ax.set_xlabel('topK (sparse budget)')
        ax.set_title(f'Context = {ctx//1000}K', fontweight='bold')
        ax.set_xlim(min(topk_values), max(topk_values))
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    axes[0].set_ylabel('Time Breakdown (%)')
    
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='center right', bbox_to_anchor=(1.15, 0.5),
              frameon=True, fancybox=True, shadow=True)
    
    fig.suptitle('KeySifter Operation Time Breakdown vs topK', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'timing_stacked_by_topk.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'timing_stacked_by_topk.pdf', bbox_inches='tight')
    print(f"  → Saved: timing_stacked_by_topk.png/pdf")
    plt.close()


def plot_heatmap(results, operations, output_dir):
    """Create heatmap showing operation percentages across all configs."""
    time_matrix, pct_matrix, configs, ctx_lengths, topk_values = extract_timing_matrix(results, operations)
    
    # Labels for x-axis
    config_labels = [f'{ctx//1000}K\ntopK={topk}' for ctx, topk, _ in configs]
    op_labels = [OPERATION_LABELS.get(op, op) for op in operations]
    
    fig, ax = plt.subplots(figsize=(max(12, len(configs) * 0.8), 6))
    
    # Custom colormap
    cmap = plt.cm.YlOrRd
    
    im = ax.imshow(pct_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=100)
    
    # Axis labels
    ax.set_xticks(np.arange(len(config_labels)))
    ax.set_yticks(np.arange(len(op_labels)))
    ax.set_xticklabels(config_labels, fontsize=9)
    ax.set_yticklabels(op_labels)
    
    # Rotate x labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add text annotations
    for i in range(len(op_labels)):
        for j in range(len(config_labels)):
            val = pct_matrix[i, j]
            if val > 0:
                text_color = 'white' if val > 50 else 'black'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center', 
                       fontsize=8, color=text_color, fontweight='bold')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Percentage of Decode Time (%)', fontsize=11)
    
    ax.set_xlabel('Configuration (Context Length, topK)', fontsize=12)
    ax.set_ylabel('Operation', fontsize=12)
    ax.set_title('KeySifter Operation Time Breakdown Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'timing_heatmap.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'timing_heatmap.pdf', bbox_inches='tight')
    print(f"  → Saved: timing_heatmap.png/pdf")
    plt.close()


def plot_trend_lines(results, operations, output_dir):
    """Create trend line plots for individual operations."""
    ctx_lengths = sorted(set(r['prompt_length'] for r in results))
    topk_values = sorted(set(r['sparse_budget'] for r in results))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Plot 1: Absolute time vs context (lines = topK)
    ax = axes[0]
    for topk in topk_values:
        filtered = sorted([r for r in results if r['sparse_budget'] == topk],
                         key=lambda r: r['prompt_length'])
        if not filtered:
            continue
        
        ctx = [r['prompt_length'] for r in filtered]
        total_time = []
        for r in filtered:
            t = sum(r['profiling_stats'][op]['mean'] * r['profiling_stats'][op]['count'] * 1000 
                   for op in operations if op in r['profiling_stats'])
            total_time.append(t)
        
        ax.plot(ctx, total_time, 'o-', label=f'topK={topk}', linewidth=2, markersize=6)
    
    ax.set_xlabel('Context Length')
    ax.set_ylabel('Total Decode Step Time (ms)')
    ax.set_title('Total Time vs Context Length', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    
    # Plot 2: TopK selection time specifically
    ax = axes[1]
    for ctx in ctx_lengths:
        filtered = sorted([r for r in results if r['prompt_length'] == ctx],
                         key=lambda r: r['sparse_budget'])
        if not filtered:
            continue
        
        topks = [r['sparse_budget'] for r in filtered]
        topk_times = []
        for r in filtered:
            s = r['profiling_stats'].get('topk_selection', {})
            topk_times.append(s.get('mean', 0) * s.get('count', 0) * 1000)
        
        ax.plot(topks, topk_times, 'o-', label=f'Ctx={ctx//1000}K', linewidth=2, markersize=6)
    
    ax.set_xlabel('topK (sparse budget)')
    ax.set_ylabel('TopK Selection Time (ms)')
    ax.set_title('TopK Selection Scaling', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Score computation time
    ax = axes[2]
    for topk in topk_values:
        filtered = sorted([r for r in results if r['sparse_budget'] == topk],
                         key=lambda r: r['prompt_length'])
        if not filtered:
            continue
        
        ctx = [r['prompt_length'] for r in filtered]
        score_times = []
        for r in filtered:
            s = r['profiling_stats'].get('compute_scores', {})
            score_times.append(s.get('mean', 0) * s.get('count', 0) * 1000)
        
        ax.plot(ctx, score_times, 'o-', label=f'topK={topk}', linewidth=2, markersize=6)
    
    ax.set_xlabel('Context Length')
    ax.set_ylabel('Score Computation Time (ms)')
    ax.set_title('Score Computation Scaling with Context', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    
    # Plot 4: Gathering time
    ax = axes[3]
    for topk in topk_values:
        filtered = sorted([r for r in results if r['sparse_budget'] == topk],
                         key=lambda r: r['prompt_length'])
        if not filtered:
            continue
        
        ctx = [r['prompt_length'] for r in filtered]
        gather_times = []
        for r in filtered:
            k_stats = r['profiling_stats'].get('get_key_cache_total', {})
            v_stats = r['profiling_stats'].get('get_value_cache_total', {})
            k_time = k_stats.get('mean', 0) * k_stats.get('count', 0)
            v_time = v_stats.get('mean', 0) * v_stats.get('count', 0)
            gather_times.append((k_time + v_time) * 1000)
        
        ax.plot(ctx, gather_times, 'o-', label=f'topK={topk}', linewidth=2, markersize=6)
    
    ax.set_xlabel('Context Length')
    ax.set_ylabel('KV Gathering Time (ms)')
    ax.set_title('KV Gathering Scaling', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    
    fig.suptitle('KeySifter Operation Scaling Trends', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'timing_trends.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'timing_trends.pdf', bbox_inches='tight')
    print(f"  → Saved: timing_trends.png/pdf")
    plt.close()


def plot_combined_figure(results, operations, output_dir):
    """Create a single combined academic figure."""
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    topk_values = sorted(set(r['sparse_budget'] for r in results))
    ctx_lengths = sorted(set(r['prompt_length'] for r in results))
    
    # Top row: Stacked area by context for different topK
    for col_idx, topk in enumerate(topk_values[:3]):  # Max 3 topK panels
        ax = fig.add_subplot(gs[0, col_idx])
        
        filtered = sorted([r for r in results if r['sparse_budget'] == topk],
                         key=lambda r: r['prompt_length'])
        if not filtered:
            continue
        
        ctx = [r['prompt_length'] for r in filtered]
        
        stacked_data = []
        for op in operations:
            times = []
            for r in filtered:
                s = r['profiling_stats'].get(op, {})
                times.append(s.get('mean', 0) * s.get('count', 0) * 1000)
            stacked_data.append(times)
        
        stacked_data = np.array(stacked_data)
        totals = stacked_data.sum(axis=0)
        totals[totals == 0] = 1  # Avoid division by zero
        pct_data = (stacked_data / totals[np.newaxis, :]) * 100
        
        colors = [COLORS.get(op, COLORS['other']) for op in operations]
        ax.stackplot(ctx, pct_data, colors=colors, alpha=0.85)
        
        ax.set_xlabel('Context Length')
        ax.set_title(f'topK = {topk:,}', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
        
        if col_idx == 0:
            ax.set_ylabel('Time Breakdown (%)')
    
    # Bottom left: Trends
    ax = fig.add_subplot(gs[1, 0])
    for topk in topk_values:
        filtered = sorted([r for r in results if r['sparse_budget'] == topk],
                         key=lambda r: r['prompt_length'])
        if not filtered:
            continue
        
        ctx = [r['prompt_length'] for r in filtered]
        total = []
        for r in filtered:
             t = sum(r['profiling_stats'].get(op, {}).get('mean', 0) * r['profiling_stats'].get(op, {}).get('count', 0) * 1000
                     for op in operations)
             total.append(t)
        ax.plot(ctx, total, 'o-', label=f'topK={topk}', linewidth=2)
    
    ax.set_xlabel('Context Length')
    ax.set_ylabel('Total Time (ms)')
    ax.set_title('Total Decode Step Time', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    
    # Bottom middle: TopK selection scaling
    ax = fig.add_subplot(gs[1, 1])
    for ctx in ctx_lengths:
        filtered = sorted([r for r in results if r['prompt_length'] == ctx],
                         key=lambda r: r['sparse_budget'])
        if not filtered:
            continue
        
        topks = [r['sparse_budget'] for r in filtered]
        topk_times = []
        for r in filtered:
            s = r['profiling_stats'].get('topk_selection', {})
            topk_times.append(s.get('mean', 0) * s.get('count', 0) * 1000)
        ax.plot(topks, topk_times, 'o-', label=f'{ctx//1000}K', linewidth=2)
    
    ax.set_xlabel('topK')
    ax.set_ylabel('TopK Selection Time (ms)')
    ax.set_title('TopK Selection Scaling', fontweight='bold')
    ax.legend(fontsize=8, title='Context')
    ax.grid(alpha=0.3)
    
    # Bottom right: Score computation scaling
    ax = fig.add_subplot(gs[1, 2])
    for topk in topk_values:
        filtered = sorted([r for r in results if r['sparse_budget'] == topk],
                         key=lambda r: r['prompt_length'])
        if not filtered:
            continue
        
        ctx = [r['prompt_length'] for r in filtered]
        score_times = []
        for r in filtered:
            s = r['profiling_stats'].get('compute_scores', {})
            score_times.append(s.get('mean', 0) * s.get('count', 0) * 1000)
        ax.plot(ctx, score_times, 'o-', label=f'topK={topk}', linewidth=2)
    
    ax.set_xlabel('Context Length')
    ax.set_ylabel('Score Computation Time (ms)')
    ax.set_title('Score Computation Scaling', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    
    # Legend for stacked areas (place outside)
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=COLORS.get(op, COLORS['other']), 
                                      edgecolor='none', alpha=0.85)
                      for op in operations]
    legend_labels = [OPERATION_LABELS.get(op, op) for op in operations]
    fig.legend(legend_elements[::-1], legend_labels[::-1], 
              loc='upper center', bbox_to_anchor=(0.5, 0.02),
              ncol=len(operations), frameon=True, fontsize=9)
    
    fig.suptitle('KeySifter Timing Breakdown: Context Length & topK Effects', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_dir / 'timing_combined.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'timing_combined.pdf', bbox_inches='tight')
    print(f"  → Saved: timing_combined.png/pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize KeySifter timing trends')
    parser.add_argument('--quick', action='store_true', help='Quick mode with fewer configs')
    parser.add_argument('--gen-length', type=int, default=256, help='Generation length per config')
    parser.add_argument('--output-dir', type=str, default='test/output', help='Output directory')
    parser.add_argument('--load-cache', type=str, help='Load cached results from JSON')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    weights_path = '/home/afa55/Projects/xKV/xKV/Llama_31_8bi_GQA_dDash8.pt'
    
    # Key operations to track
    operations = [
        'predictor_forward',
        'compute_scores',
        'topk_selection',
        'get_key_cache_total',
        'get_value_cache_total',
        'update_kv_cache_total',
    ]
    
    if args.load_cache:
        print(f"Loading cached results from {args.load_cache}")
        with open(args.load_cache, 'r') as f:
            results = json.load(f)
    else:
        if args.quick:
            context_lengths = [4096, 8192, 16384]
            topk_values = [512, 1024, 2048]
        else:
            context_lengths = [4096, 8192, 16384, 32768, 65536]
            topk_values = [1024, 2048, 4096]
        
        results = collect_sweep_data(context_lengths, topk_values, 
                                     args.gen_length, weights_path)
        
        # import pdb; pdb.set_trace()
        # Save results
        cache_file = output_dir / 'timing_sweep_results.json'
        with open(cache_file, 'w') as f:
            # Convert for JSON serialization
            json_results = []
            for r in results:
                jr = {
                    'prompt_length': r['prompt_length'],
                    'sparse_budget': r['sparse_budget'],
                    'profiling_stats': {
                        op: {k: float(v) for k, v in stats.items()}
                        for op, stats in r['profiling_stats'].items()
                    }
                }
                json_results.append(jr)
            json.dump(json_results, f, indent=2)
        print(f"\n✓ Cached results to {cache_file}")
    
    # Create visualizations
    print(f"\n{'='*60}")
    print("Creating visualizations...")
    print(f"{'='*60}")
    
    plot_stacked_area_by_context(results, operations, output_dir)
    plot_stacked_area_by_topk(results, operations, output_dir)
    plot_heatmap(results, operations, output_dir)
    plot_trend_lines(results, operations, output_dir)
    plot_combined_figure(results, operations, output_dir)
    
    print(f"\n{colored('✓ All visualizations saved to ' + str(output_dir), 'green')}")


if __name__ == '__main__':
    main()
