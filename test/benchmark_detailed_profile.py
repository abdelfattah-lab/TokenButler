#!/usr/bin/env python3
"""
Detailed profiling of KeySifter decode operations.
Instruments individual components to identify optimization priorities.
"""

import torch
import time
import sys
import gc
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
sys.path.insert(0, '/home/afa55/Projects/xKV/xKV')

from models import Llama
from termcolor import colored


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
    
    # Store original methods
    original_compute_predictor = llm.kv_cache.compute_predictor_importance
    original_get_retrieval = llm.kv_cache.get_retrieval_position_ids
    original_get_key = llm.kv_cache.get_key_cache
    original_get_value = llm.kv_cache.get_value_cache
    original_update_kv = llm.kv_cache.update_kv_cache
    
    # Wrap compute_predictor_importance
    def profiled_compute_predictor(hidden_states, producer_layer_idx):
        with profiler.record('predictor_forward'):
            return original_compute_predictor(hidden_states, producer_layer_idx)
    
    # Wrap get_retrieval_position_ids with detailed breakdown
    def profiled_get_retrieval(layer_idx, query_states):
        with profiler.record('get_retrieval_total'):
            bsz = query_states.shape[0]
            slot_idx = layer_idx % llm.kv_cache.producer_frequency
            
            # Get importance query
            with profiler.record('get_importance_query'):
                q_slot = llm.kv_cache.q_importance_cache[:, slot_idx, :, :]
                Lq = q_slot.shape[1]
                q_slot = q_slot.view(bsz, llm.kv_cache.num_attention_heads, Lq, llm.kv_cache.dDash)
                q_slot = q_slot[:, :, -1:, :]
                q_slot = q_slot.view(bsz, llm.kv_cache.num_key_value_heads, llm.kv_cache.num_key_value_groups, 1, llm.kv_cache.dDash)
            
            # Get projected keys
            with profiler.record('get_projected_keys'):
                k_proj = llm.kv_cache.k_proj_cache[layer_idx, :, :, :llm.kv_cache.last_projected_pos]
                k_proj = k_proj.unsqueeze(2)
            
            # Compute scores
            with profiler.record('compute_scores'):
                scores = torch.einsum("bhgqd,bhgkd->bhgqk", q_slot, k_proj)
                scores = scores.squeeze(3) / np.sqrt(llm.kv_cache.dDash)
                scores = scores.max(dim=2).values
            
            # Mask local window
            with profiler.record('mask_local_window'):
                local_start = max(0, llm.kv_cache.kv_offset - llm.kv_cache.local_window)
                if local_start < llm.kv_cache.last_projected_pos:
                    scores[:, :, local_start:] = float("-inf")
            
            # TopK selection
            with profiler.record('topk_selection'):
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
            
            return position_ids
    
    # Wrap get_key_cache
    def profiled_get_key(layer_idx, position_ids, rope_func=None, cos_sin_cache=None):
        with profiler.record('get_key_cache_total'):
            with profiler.record('gather_sparse_keys'):
                result = original_get_key(layer_idx, position_ids, rope_func, cos_sin_cache)
            return result
    
    # Wrap get_value_cache
    def profiled_get_value(layer_idx, position_ids):
        with profiler.record('get_value_cache_total'):
            with profiler.record('gather_sparse_values'):
                result = original_get_value(layer_idx, position_ids)
            return result
    
    # Wrap update_kv_cache
    def profiled_update_kv(new_k_cache, new_v_cache, layer_idx):
        with profiler.record('update_kv_cache_total'):
            with profiler.record('buffer_update'):
                result = original_update_kv(new_k_cache, new_v_cache, layer_idx)
            return result
    
    # Apply monkey patches
    llm.kv_cache.compute_predictor_importance = profiled_compute_predictor
    llm.kv_cache.get_retrieval_position_ids = profiled_get_retrieval
    llm.kv_cache.get_key_cache = profiled_get_key
    llm.kv_cache.get_value_cache = profiled_get_value
    llm.kv_cache.update_kv_cache = profiled_update_kv


def benchmark_keysifter_detailed(prompt_length, gen_length, sparse_budget=1024, predictor_path=''):
    """Benchmark KeySifter with detailed component breakdown."""
    print(f"\n{'='*60}")
    print(f"Detailed KeySifter Profiling | Prompt: {prompt_length} tokens | Gen: {gen_length} tokens")
    print(f"{'='*60}")

    # Initialize model
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
        'dDash': 16,
        'producer_frequency': 4,
        'keysifter_intermediate_dim': 1024,
        'predictor_path': predictor_path,
    }

    print("Initializing model...")
    llm = Llama(**model_kwargs)
    
    # Create profiler and instrument
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
    print(f"Actual prompt length: {actual_prompt_len} tokens")

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Prefill
    print("\n[PREFILL]")
    start = time.perf_counter()
    torch.cuda.synchronize()
    logits = llm.prefill(input_ids)
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - start
    print(f"  Time: {prefill_time:.4f}s")

    # KV cache H2D
    h2d_start = torch.cuda.Event(enable_timing=True)
    h2d_end = torch.cuda.Event(enable_timing=True)
    h2d_start.record()
    llm.kv_cache.H2D()
    h2d_end.record()
    torch.cuda.synchronize()
    kv_h2d_time = h2d_start.elapsed_time(h2d_end) / 1000.0
    print(f"\n[KV H2D Transfer]: {kv_h2d_time:.4f}s")

    # Decode with profiling
    print(f"\n[DECODE - Profiling {gen_length} steps]")
    
    # Sample every N steps for lower overhead
    sample_rate = max(1, gen_length // 100)
    sampled_steps = 0
    
    for i in range(gen_length):
        torch.cuda.synchronize()
        next_token = logits.argmax(dim=-1)
        
        # Only profile sampled steps
        profiler.active = (i % sample_rate == 0)
        if profiler.active:
            sampled_steps += 1
        
        # Time overall step
        step_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        step_start.record()
        
        position_ids = llm.get_ctx(next_token)
        logits = llm.inference(input_ids=next_token, position_ids=position_ids)
        
        step_end.record()
        torch.cuda.synchronize()
        
        if i % 100 == 0:
            step_time = step_start.elapsed_time(step_end)
            print(f"  Step {i}: {step_time:.2f}ms")
    
    print(f"\n  Profiled {sampled_steps} steps (every {sample_rate})")

    # Get statistics
    stats = profiler.get_stats()
    
    # Memory
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"\n[MEMORY]: {memory_allocated:.2f} GB")

    # Cleanup
    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'prompt_length': actual_prompt_len,
        'gen_length': gen_length,
        'prefill_time': prefill_time,
        'kv_h2d_time': kv_h2d_time,
        'profiling_stats': stats,
        'memory_allocated_gb': memory_allocated,
        'sampled_steps': sampled_steps,
    }


def plot_detailed_breakdown(result, output_file='decode_detailed_breakdown.png'):
    """Create visualization of detailed component breakdown."""
    stats = result['profiling_stats']
    
    if not stats:
        print("No profiling data available")
        return
    
    # Define operation hierarchy and colors
    # Top-level categories
    categories = {
        'Predictor': ['predictor_forward'],
        'Index Selection': ['get_retrieval_total', 'get_importance_query', 'get_projected_keys', 
                           'compute_scores', 'mask_local_window', 'topk_selection'],
        'KV Gathering': ['get_key_cache_total', 'gather_sparse_keys', 
                         'get_value_cache_total', 'gather_sparse_values'],
        'Cache Update': ['update_kv_cache_total', 'buffer_update'],
    }
    
    # Colors for categories
    cat_colors = {
        'Predictor': '#FF6B6B',
        'Index Selection': '#4ECDC4',
        'KV Gathering': '#FFD93D',
        'Cache Update': '#95E1D3',
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Plot 1: Top-level breakdown ---
    top_level_times = {}
    for cat_name, ops in categories.items():
        # Use the first (most general) operation if available, else sum sub-operations
        primary_op = ops[0]
        if primary_op in stats:
            top_level_times[cat_name] = stats[primary_op]['mean'] * 1000  # Convert to ms
        else:
            # Sum sub-operations
            total = sum(stats[op]['mean'] * 1000 for op in ops[1:] if op in stats)
            if total > 0:
                top_level_times[cat_name] = total
    
    # Sort by time
    sorted_cats = sorted(top_level_times.items(), key=lambda x: x[1], reverse=True)
    cat_names = [c[0] for c in sorted_cats]
    cat_times = [c[1] for c in sorted_cats]
    colors = [cat_colors.get(c, '#CCCCCC') for c in cat_names]
    
    total_time = sum(cat_times)
    
    bars = ax1.bar(cat_names, cat_times, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, time_val in zip(bars, cat_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
               f'{time_val:.2f}ms\n({time_val/total_time*100:.1f}%)',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.axhline(y=total_time, color='red', linestyle='--', linewidth=2, alpha=0.7, 
               label=f'Total: {total_time:.2f}ms')
    ax1.set_title('Top-Level Component Breakdown', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Time per Decode Step (ms)', fontsize=11)
    ax1.set_ylim(0, total_time * 1.2)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # --- Plot 2: Detailed sub-operations ---
    # Show all operations with significant time
    all_ops = {}
    for op_name, op_stats in stats.items():
        time_ms = op_stats['mean'] * 1000
        if time_ms > 0.01:  # Only show ops > 0.01ms
            all_ops[op_name] = time_ms
    
    sorted_ops = sorted(all_ops.items(), key=lambda x: x[1], reverse=True)
    op_names = [o[0].replace('_', ' ').title() for o in sorted_ops[:15]]  # Top 15
    op_times = [o[1] for o in sorted_ops[:15]]
    
    # Color by category
    op_colors = []
    for op_full in [o[0] for o in sorted_ops[:15]]:
        color = '#CCCCCC'
        for cat_name, ops in categories.items():
            if op_full in ops:
                color = cat_colors[cat_name]
                break
        op_colors.append(color)
    
    bars2 = ax2.barh(op_names, op_times, color=op_colors, edgecolor='black', linewidth=1)
    
    for bar, time_val in zip(bars2, op_times):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
               f' {time_val:.3f}ms',
               ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Time per Decode Step (ms)', fontsize=11, fontweight='bold')
    ax2.set_title('Detailed Operation Breakdown (Top 15)', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()
    
    plt.suptitle(f'KeySifter Detailed Profiling - {result["prompt_length"]:,} token context',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n{colored('✓ Detailed breakdown saved to ' + output_file, 'green')}")
    plt.close()


def print_detailed_analysis(result):
    """Print detailed text analysis."""
    stats = result['profiling_stats']
    
    print("\n" + "="*80)
    print(colored("DETAILED COMPONENT ANALYSIS", 'cyan', attrs=['bold']))
    print("="*80)
    
    # Group by category
    categories = {
        'Predictor': ['predictor_forward'],
        'Index Selection': ['get_retrieval_total', 'get_importance_query', 'get_projected_keys', 
                           'compute_scores', 'mask_local_window', 'topk_selection'],
        'KV Gathering': ['get_key_cache_total', 'gather_sparse_keys', 
                         'get_value_cache_total', 'gather_sparse_values'],
        'Cache Update': ['update_kv_cache_total', 'buffer_update'],
    }
    
    total_time = 0
    cat_times = {}
    
    for cat_name, ops in categories.items():
        # Use the first (most general) operation if available, else sum sub-operations
        # This matches the plot logic to avoid double-counting
        primary_op = ops[0]
        if primary_op in stats:
            cat_times[cat_name] = stats[primary_op]['mean'] * 1000
            total_time += stats[primary_op]['mean'] * 1000
        else:
            # Sum sub-operations only if primary not available
            cat_total = sum(stats[op]['mean'] * 1000 for op in ops[1:] if op in stats)
            if cat_total > 0:
                cat_times[cat_name] = cat_total
                total_time += cat_total
    
    print(f"\nPrompt: {result['prompt_length']:,} tokens | Generated: {result['gen_length']} tokens")
    print(f"Sampled steps: {result['sampled_steps']}")
    print("-" * 80)
    
    # Print by category
    for cat_name in sorted(cat_times.keys(), key=lambda x: cat_times[x], reverse=True):
        time_ms = cat_times[cat_name]
        pct = time_ms / total_time * 100
        print(f"\n{colored(cat_name, 'yellow', attrs=['bold'])}: {time_ms:.3f}ms ({pct:.1f}%)")
        
        # Print sub-operations
        # Note: The first operation is the total, sub-operations show breakdown within it
        cat_ops = categories[cat_name]
        primary_op = cat_ops[0]
        
        # Show primary (total) operation
        if primary_op in stats:
            op_time = stats[primary_op]['mean'] * 1000
            op_pct = op_time / total_time * 100
            op_std = stats[primary_op]['std'] * 1000
            print(f"  {primary_op:30s}: {op_time:7.3f}ms ({op_pct:5.1f}%) ±{op_std:.3f}ms [TOTAL]")
            
            # Show sub-operations with percentage relative to the category total
            for op in cat_ops[1:]:
                if op in stats:
                    op_time = stats[op]['mean'] * 1000
                    op_pct_of_cat = op_time / time_ms * 100  # Percentage of category, not overall
                    op_pct_of_total = op_time / total_time * 100
                    op_std = stats[op]['std'] * 1000
                    print(f"    ├─ {op:28s}: {op_time:7.3f}ms ({op_pct_of_cat:5.1f}% of {cat_name}) ±{op_std:.3f}ms")
        else:
            # No primary operation, just list sub-operations
            for op in cat_ops[1:]:
                if op in stats:
                    op_time = stats[op]['mean'] * 1000
                    op_pct = op_time / total_time * 100
                    op_std = stats[op]['std'] * 1000
                    print(f"  {op:30s}: {op_time:7.3f}ms ({op_pct:5.1f}%) ±{op_std:.3f}ms")
    
    print(f"\n{'─'*80}")
    print(f"TOTAL (KeySifter overhead):    {total_time:7.3f}ms (100.0%)")
    
    # Optimization recommendations
    print(f"\n{colored('💡 Optimization Priority:', 'cyan', attrs=['bold'])}")
    sorted_cats = sorted(cat_times.items(), key=lambda x: x[1], reverse=True)
    for i, (cat_name, time_ms) in enumerate(sorted_cats[:5], 1):
        pct = time_ms / total_time * 100
        print(f"  {i}. {cat_name:20s} - {time_ms:7.3f}ms ({pct:5.1f}%)")


def main():
    print(colored("="*80, 'cyan'))
    print(colored("KeySifter Detailed Component Profiling", 'cyan', attrs=['bold']))
    print(colored("="*80, 'cyan'))

    # Configuration
    configs = [
        (16384, 8192),   # Medium test
        # (65536, 1024),  # Larger test (uncomment if needed)
    ]

    weights_path = '/home/afa55/Projects/xKV/xKV/Llama_31_8bi_p4x.pt'
    
    for prompt_len, gen_len in configs:
        result = benchmark_keysifter_detailed(prompt_len, gen_len, 
                                              sparse_budget=1024, 
                                              predictor_path=weights_path)
        
        # Print analysis
        print_detailed_analysis(result)
        
        # Create visualization
        plot_detailed_breakdown(result)
        
        # Save results
        # Convert numpy types to native Python for JSON
        json_result = {
            'prompt_length': result['prompt_length'],
            'gen_length': result['gen_length'],
            'prefill_time': result['prefill_time'],
            'kv_h2d_time': result['kv_h2d_time'],
            'memory_allocated_gb': result['memory_allocated_gb'],
            'sampled_steps': result['sampled_steps'],
            'profiling_stats': {
                op: {k: float(v) for k, v in stats.items()}
                for op, stats in result['profiling_stats'].items()
            }
        }
        
        with open('decode_detailed_profile.json', 'w') as f:
            json.dump(json_result, f, indent=2)
        print(f"\n{colored('✓ Results saved to decode_detailed_profile.json', 'green')}")


if __name__ == '__main__':
    main()
