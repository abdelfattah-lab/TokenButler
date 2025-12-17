#!/usr/bin/env python3
"""
Visualize KeySifter absolute timing breakdown trends across context lengths.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
from pathlib import Path
from termcolor import colored

# Add parent directory to path to allow importing from sibling script
sys.path.append(str(Path(__file__).parent))

from visualize_timing_trends import (
    collect_sweep_data,
    COLORS, 
    OPERATION_LABELS,
    DetailedProfiler,
    monkey_patch_keysifter_for_profiling
)

# Add colors for new operations
COLORS['flash_attn_compute'] = '#D55E00'      # Vermilion
COLORS['qkv_projection'] = '#56B4E9'          # Sky Blue
COLORS['rope_embedding'] = '#0072B2'          # Blueish Green
COLORS['mlp_compute'] = '#F0E442'             # Yellow
COLORS['prepare_tensors'] = '#AA4499'         # Purple (new)

import torch
import gc
from models import Llama
import models.base



# Academic-quality plot settings (copied to ensure consistency)
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

def run_single_benchmark_with_total(prompt_length, gen_length, sparse_budget, predictor_path):
    """Run benchmark for a single configuration, measuring TOTAL step time."""
    print(f"\n  → Context: {prompt_length:,} | topK: {sparse_budget:,}")
    
    model_kwargs = {
        'model_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'batch_size': 1,
        'max_length': 136072,  # Large enough
        'device': 'cuda:0',
        'dtype': torch.bfloat16,
        'attn_mode': 'keysifter',
        'sparse_budget': sparse_budget,
        'chunk_size': 8,
        'rank': 160,
        'dDash': 16,
        'producer_frequency': 4,
        'keysifter_intermediate_dim': 1024,
        'predictor_path': predictor_path,
    }

    llm = Llama(**model_kwargs)
    profiler = DetailedProfiler()
    profiler.active = False # Ensure not profiling prefill
    monkey_patch_keysifter_for_profiling(llm, profiler)
    
    
    # --- Comprehensive Monkey Patching ---
    
    # 1. Flash Attention (already prepared)
    original_flash_attn = models.base.flash_attn_with_kvcache
    def profiled_flash_attn(*args, **kwargs):
        with profiler.record('flash_attn_compute'):
            return original_flash_attn(*args, **kwargs)
    models.base.flash_attn_with_kvcache = profiled_flash_attn

    # 2. QKV Projection (Pre-Attention)
    original_pre_attn = llm.pre_attention_compute
    def profiled_pre_attn(*args, **kwargs):
        with profiler.record('qkv_projection'):
            return original_pre_attn(*args, **kwargs)
    llm.pre_attention_compute = profiled_pre_attn

    # 3. RoPE
    original_rope = llm.apply_rotary_pos_emb
    def profiled_rope(*args, **kwargs):
        with profiler.record('rope_embedding'):
            return original_rope(*args, **kwargs)
    llm.apply_rotary_pos_emb = profiled_rope

    # 4. MLP & Output (Post-Attention)
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
    sample_rate = max(1, gen_length // 50)
    
    for i in range(gen_length):
        torch.cuda.synchronize()
        next_token = logits.argmax(dim=-1)
        
        # Only activate profiler for sampled steps to reduce overhead
        profiler.active = (i % sample_rate == 0)
        
        position_ids = llm.get_ctx(next_token)
        
        # Wrap the ENTIRE inference step
        with profiler.record('total_step_time'):
            logits = llm.inference(input_ids=next_token, position_ids=position_ids)
        
        torch.cuda.synchronize()

    stats = profiler.get_stats()

    # Restore original functions
    models.base.flash_attn_with_kvcache = original_flash_attn
    # Note: Instance methods (llm.x) don't need class restoration if we just delete llm, 
    # but good practice if we were keeping the object. Here we del llm.
    # However, because we assigned to the instance `llm.method = ...`, 
    # we don't strictly need to revert class methods unless we patched the class.
    # Here we patched the instance methods mostly (except flash_attn which is module level).
    
    del llm
    
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'prompt_length': actual_prompt_len,
        'sparse_budget': sparse_budget,
        'profiling_stats': stats,
    }

def collect_sweep_data_with_total(context_lengths, topk_values, gen_length, predictor_path):
    """Run benchmarks across all configurations using the new total-aware runner."""
    results = []
    
    total_configs = len(context_lengths) * len(topk_values)
    print(f"\n{'='*60}")
    print(f"Running {total_configs} configurations (measuring Total Time)...")
    print(f"{'='*60}")
    
    for ctx_len in context_lengths:
        for topk in topk_values:
            if topk > ctx_len // 2:
                print(f"  → Skipping topK={topk} > context/2 for ctx={ctx_len}")
                continue
            
            result = run_single_benchmark_with_total(ctx_len, gen_length, topk, predictor_path)
            results.append(result)
    
    return results


def plot_stacked_area_by_context_absolute(results, operations, output_dir):
    """Create stacked area chart: ABSOLUTE time breakdown vs context length, faceted by topK."""
    topk_values = sorted(set(r['sparse_budget'] for r in results))
    
    n_topk = len(topk_values)
    fig, axes = plt.subplots(1, n_topk, figsize=(4.5 * n_topk, 5), sharey=True)
    if n_topk == 1:
        axes = [axes]
    
    # Track max Y for setting consistent limits if desired, or let them float
    # For comparison, fixed limits might be better, but let's see range first.
    # We will share Y axis so they match automatically.
    
    for ax_idx, topk in enumerate(topk_values):
        ax = axes[ax_idx]
        
        # Filter results for this topK
        filtered = [r for r in results if r['sparse_budget'] == topk]
        filtered = sorted(filtered, key=lambda r: r['prompt_length'])
        
        if not filtered:
            continue
        
        ctx_lengths = [r['prompt_length'] for r in filtered]
        
        # Build stacked data (Absolute times in ms)
        stacked_data = []
        for op in operations:
            times = []
            for r in filtered:
                # Get step count for THIS run
                if 'total_step_time' in r['profiling_stats']:
                    s_total = r['profiling_stats']['total_step_time']
                    step_count = max(1, s_total['count'])
                    total_token_ms = s_total['mean'] * 1000
                else:
                    step_count = 1
                    total_token_ms = 0

                if op == 'other_model_ops':
                    # Sum of all explicitly tracked ops for this run
                    trackable_sum_ms = 0
                    for other_op in operations:
                        if other_op != 'other_model_ops' and other_op in r['profiling_stats']:
                            s_op = r['profiling_stats'][other_op]
                            # Time per token = (mean * count) / step_count
                            trackable_sum_ms += (s_op['mean'] * s_op['count'] * 1000) / step_count
                    
                    other_ms = max(0, total_token_ms - trackable_sum_ms)
                    times.append(other_ms)
                
                elif op in r['profiling_stats']:
                    s = r['profiling_stats'][op]
                    # Time per token = (mean * count) / step_count
                    time_per_token = (s['mean'] * s['count'] * 1000) / step_count
                    times.append(time_per_token)
                else:
                    times.append(0)
            stacked_data.append(times)
        
        stacked_data = np.array(stacked_data)
        
        # Plot stacked area
        # Ensure we have a color for 'other_model_ops'
        local_colors = [COLORS.get(op, '#DDDDDD') if op != 'other_model_ops' else '#DDDDDD' for op in operations]
        
        labels_map = OPERATION_LABELS.copy()
        labels_map['other_model_ops'] = 'Other (Overhead)'
        labels_map['flash_attn_compute'] = 'Attention Kernel'
        labels_map['qkv_projection'] = 'QKV Projection'
        labels_map['rope_embedding'] = 'RoPE'
        labels_map['mlp_compute'] = 'MLP & Output'
        labels_map['prepare_tensors'] = 'Tensor Prep'
        
        ax.stackplot(ctx_lengths, stacked_data, labels=[labels_map.get(op, op) for op in operations],
                    colors=local_colors, alpha=0.85)
        
        ax.set_xlabel('Context Length (tokens)')
        ax.set_title(f'topK = {topk:,}', fontweight='bold')
        ax.set_xlim(min(ctx_lengths), max(ctx_lengths))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Format x-axis with K suffix
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    
    axes[0].set_ylabel('Time per Token (ms)')
    
    # Legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='center right', bbox_to_anchor=(1.15, 0.5),
              frameon=True, fancybox=True, shadow=True)
    
    fig.suptitle('KeySifter Time per Token Breakdown vs Context Length', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = output_dir / 'timing_stacked_by_context.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"  → Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize KeySifter absolute timing trends')
    parser.add_argument('--quick', action='store_true', help='Quick mode with fewer configs')
    parser.add_argument('--gen-length', type=int, default=1024, help='Generation length per config')
    parser.add_argument('--output-dir', type=str, default='test/output', help='Output directory')
    parser.add_argument('--load-cache', type=str, help='Load cached results from JSON')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    weights_path = '/home/afa55/Projects/xKV/xKV/Llama_31_8bi_GQA_dDash16.pt'
    
    operations = [
        'qkv_projection',  # Pre-attention QKV projection
        'predictor_forward',  # KeySifter predictor forward pass
        'prepare_tensors',  # Tensor reshaping for batched operations
        'compute_scores',  # Score computation via einsum
        'topk_selection',  # TopK selection
        'get_key_cache_total',  # Key gathering
        'get_value_cache_total',  # Value gathering
        'rope_embedding',  # RoPE embedding
        'update_kv_cache_total',  # KV cache update (not currently instrumented in KeySifterCache)
        'flash_attn_compute',  # Flash attention kernel
        'mlp_compute',  # MLP/FFN computation
        'other_model_ops',  # Residual overhead
    ]
    
    results = None
    if args.load_cache and Path(args.load_cache).exists():
        print(f"Loading cached results from {args.load_cache}")
        with open(args.load_cache, 'r') as f:
            results = json.load(f)
    elif args.load_cache:
        print(f"Warning: Cache file {args.load_cache} not found. Running benchmark.")
        
    if results is None: 
        if args.quick:
            context_lengths = [4096, 8192]
            topk_values = [1024]
        else:
            context_lengths = [4096, 8192, 16384, 32768, 65536, 131072]
            topk_values = [1024, 2048, 4096, 8192]
        
        # Use new collection function
        results = collect_sweep_data_with_total(context_lengths, topk_values, 
                                     args.gen_length, weights_path)
                                     
        # Save results
        cache_file = output_dir / 'timing_sweep_results_absolute.json'
        with open(cache_file, 'w') as f:
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

    print(f"\n{'='*60}")
    print("Creating visualization...")
    print(f"{'='*60}")
    
    plot_stacked_area_by_context_absolute(results, operations, output_dir)
    print(f"\n{colored('✓ Visualization saved to ' + str(output_dir), 'green')}")

if __name__ == '__main__':
    main()
