#!/usr/bin/env python3
"""
Benchmark KeySifter vs Full Attention for runtime efficiency.
Tests prefill and decode performance across different sequence lengths.
"""

import torch
import time
import sys
import gc
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, '/home/afa55/Projects/xKV/xKV')

# Suppress dynamo errors/warnings to avoid spam from skipped CUDAGraphs
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from models import Llama
from termcolor import colored

def benchmark_model(attn_mode, prompt_length, gen_length, sparse_budget=512, predictor_path='', oracle_random_indices=True):
    """Benchmark a single configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {attn_mode.upper()} | Prompt: {prompt_length} tokens | Gen: {gen_length} tokens")
    print(f"{'='*60}")

    # Initialize model
    model_kwargs = {
        'model_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'batch_size': 1,
        'max_length': 136072,
        'device': 'cuda:0',
        'dtype': torch.bfloat16,
        'attn_mode': attn_mode,
    }

    if attn_mode == 'keysifter':
        model_kwargs.update({
            'sparse_budget': sparse_budget,
            'chunk_size': 8,
            'rank': 160,
            'dDash': 16,
            'producer_frequency': 4,
            'keysifter_intermediate_dim': 1024,
            'predictor_path': predictor_path,
        })
    elif attn_mode == 'oracle':
        model_kwargs.update({
            'sparse_budget': sparse_budget,
            'chunk_size': 8,
            'dDash': 16,  # Or whatever default
            'oracle_random_indices': oracle_random_indices,
        })
    elif attn_mode == 'shadowkv':
        model_kwargs.update({
            'sparse_budget': sparse_budget,
            'chunk_size': 8,
            'rank': 160,
        })

    print("Initializing model...")
    llm = Llama(**model_kwargs)
    


    # Create prompt
    # Use repeated text to reach desired length
    base_text = "The quick brown fox jumps over the lazy dog. "
    repetitions = (prompt_length // 10) + 1  # ~10 tokens per repetition
    text = base_text * repetitions
    input_ids = llm.encode(text)

    # Truncate to exact length
    if input_ids.shape[1] > prompt_length:
        input_ids = input_ids[:, :prompt_length]

    actual_prompt_len = input_ids.shape[1]
    print(f"Actual prompt length: {actual_prompt_len} tokens")

    # Clear cache and warmup
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Benchmark prefill
    print("\n[PREFILL]")
    start = time.perf_counter()
    torch.cuda.synchronize()

    logits = llm.prefill(input_ids)

    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - start

    prefill_tokens_per_sec = actual_prompt_len / prefill_time
    print(f"  Time: {prefill_time:.4f}s")
    print(f"  Throughput: {prefill_tokens_per_sec:.2f} tokens/s")

    if False:
        # Apply torch.compile AFTER prefill to optimize decode runtime (max-autotune)
        # This avoids CUDAGraphs errors during prefill chunking loops
        try:
            print(f"Compiling model components with torch.compile (max-autotune) for decode...")
            # Compile layer_compute (most critical)
            llm.layer_compute = torch.compile(llm.layer_compute, mode="max-autotune")
            llm.pre_attention_compute = torch.compile(llm.pre_attention_compute, mode="max-autotune") 
            llm.post_attention_compute = torch.compile(llm.post_attention_compute, mode="max-autotune")
        except Exception as e:
            print(f"Compilation warning: {e}")

    # Benchmark decode (sampled timings)
    print("\n[DECODE]")
    # Time the Host->Device transfer for KV cache
    try:
        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        h2d_start.record()
        llm.kv_cache.H2D()
        h2d_end.record()
        torch.cuda.synchronize()
        kv_h2d_time = h2d_start.elapsed_time(h2d_end) / 1000.0
        print(f"  kv_cache.H2D time: {kv_h2d_time:.4f}s")
    except Exception:
        # Fallback if CUDA events not available
        t0 = time.perf_counter()
        llm.kv_cache.H2D()
        torch.cuda.synchronize()
        kv_h2d_time = time.perf_counter() - t0
        print(f"  kv_cache.H2D time (fallback): {kv_h2d_time:.4f}s")

    # We'll keep a low-overhead wall-clock measurement per-step, and
    # also take sampled CUDA event timings every `sample_rate` steps
    decode_times = []             # wall-clock per-step (all steps)
    sampled_total = []            # sampled total time (CUDA events) seconds
    sampled_getctx = []           # sampled get_ctx time
    sampled_inference = []        # sampled inference time

    sample_rate = max(1, int(gen_length // 50))  # ~50 samples across generation

    for i in range(gen_length):
        # wall-clock start
        torch.cuda.synchronize()
        # Mark step begin for CUDAGraphs (max-autotune) reliability
        torch.compiler.cudagraph_mark_step_begin()
        
        start = time.perf_counter()

        next_token = logits.argmax(dim=-1)

        # Sample detailed timings periodically to avoid high overhead
        if (i % sample_rate) == 0:
            evt_total_s = torch.cuda.Event(enable_timing=True)
            evt_total_e = torch.cuda.Event(enable_timing=True)
            evt_get_s = torch.cuda.Event(enable_timing=True)
            evt_get_e = torch.cuda.Event(enable_timing=True)
            evt_inf_s = torch.cuda.Event(enable_timing=True)
            evt_inf_e = torch.cuda.Event(enable_timing=True)

            evt_total_s.record()

            evt_get_s.record()
            position_ids = llm.get_ctx(next_token)
            evt_get_e.record()

            evt_inf_s.record()
            logits = llm.inference(input_ids=next_token, position_ids=position_ids)
            evt_inf_e.record()

            evt_total_e.record()
            torch.cuda.synchronize()

            total_ms = evt_total_s.elapsed_time(evt_total_e)
            get_ms = evt_get_s.elapsed_time(evt_get_e)
            inf_ms = evt_inf_s.elapsed_time(evt_inf_e)

            sampled_total.append(total_ms / 1000.0)
            sampled_getctx.append(get_ms / 1000.0)
            sampled_inference.append(inf_ms / 1000.0)
        else:
            position_ids = llm.get_ctx(next_token)
            logits = llm.inference(input_ids=next_token, position_ids=position_ids)

        torch.cuda.synchronize()
        step_time = time.perf_counter() - start
        decode_times.append(step_time)

    # Compute averages
    avg_decode_time = sum(decode_times) / len(decode_times)
    decode_tokens_per_sec = 1.0 / avg_decode_time

    sample_info = {}
    if sampled_total:
        sample_info = {
            'sample_count': len(sampled_total),
            'avg_sample_total_sec': sum(sampled_total) / len(sampled_total),
            'avg_sample_getctx_sec': sum(sampled_getctx) / len(sampled_getctx),
            'avg_sample_inference_sec': sum(sampled_inference) / len(sampled_inference),
        }

    print(f"  Avg time per token: {avg_decode_time*1000:.2f}ms")
    print(f"  Throughput: {decode_tokens_per_sec:.2f} tokens/s")
    print(f"  Min/Max: {min(decode_times)*1000:.2f}ms / {max(decode_times)*1000:.2f}ms")
    if sample_info:
        print(f"  Sampled ({sample_info['sample_count']}): total {sample_info['avg_sample_total_sec']*1000:.2f}ms | get_ctx {sample_info['avg_sample_getctx_sec']*1000:.2f}ms | inference {sample_info['avg_sample_inference_sec']*1000:.2f}ms")

    # Total time
    total_time = prefill_time + sum(decode_times)
    print(f"\n[TOTAL]")
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Overall throughput: {(actual_prompt_len + gen_length) / total_time:.2f} tokens/s")

    # Memory usage
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"\n[MEMORY]")
    print(f"  Allocated: {memory_allocated:.2f} GB")
    print(f"  Reserved: {memory_reserved:.2f} GB")

    # Cleanup
    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'attn_mode': attn_mode,
        'prompt_length': actual_prompt_len,
        'gen_length': gen_length,
        'prefill_time': prefill_time,
        'prefill_tokens_per_sec': prefill_tokens_per_sec,
        'decode_time_avg': avg_decode_time,
        'decode_tokens_per_sec': decode_tokens_per_sec,
        'total_time': total_time,
        'memory_allocated_gb': memory_allocated,
        'kv_h2d_time': kv_h2d_time,
        'sample_info': sample_info,
    }


def plot_decode_breakdown(results_list, output_file='decode_breakdown.png'):
    """Plot decode stage time breakdown for KeySifter."""
    # Filter for KeySifter or Oracle results
    # Oracle results might not have detailed sample info if we skipped predictor, 
    # but let's see if we added sample collection logic for it. The benchmark_model code collects samples regardless of mode.
    # So both should have sample_info.
    keysifter_results = [r for r in results_list 
                         if (r['attn_mode'] == 'keysifter' or r['attn_mode'] == 'oracle') and r.get('sample_info')]
    
    if not keysifter_results:
        print("No KeySifter results with timing breakdown found.")
        return
    
    # Create figure with subplots
    n_configs = len(keysifter_results)
    fig, axes = plt.subplots(1, n_configs, figsize=(6*n_configs, 5))
    if n_configs == 1:
        axes = [axes]
    
    for idx, result in enumerate(keysifter_results):
        ax = axes[idx]
        sample_info = result['sample_info']
        
        # Extract timing components (in milliseconds)
        get_ctx_ms = sample_info['avg_sample_getctx_sec'] * 1000
        inference_ms = sample_info['avg_sample_inference_sec'] * 1000
        total_ms = sample_info['avg_sample_total_sec'] * 1000
        
        # Calculate overhead (total - get_ctx - inference)
        overhead_ms = max(0, total_ms - get_ctx_ms - inference_ms)
        
        # Also include KV H2D transfer time
        h2d_ms = result.get('kv_h2d_time', 0) * 1000
        
        # Create bar chart
        components = ['get_ctx\n(KeySifter)', 'inference\n(Model)', 'overhead']
        times = [get_ctx_ms, inference_ms, overhead_ms]
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        
        bars = ax.bar(components, times, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_val:.2f}ms\n({time_val/total_ms*100:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add total time annotation
        ax.axhline(y=total_ms, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Total: {total_ms:.2f}ms')
        
        # Formatting
        prompt_len = result['prompt_length']
        gen_len = result['gen_length']
        ax.set_title(f'Decode Stage Breakdown\nPrompt: {prompt_len:,} tokens | Gen: {gen_len} tokens',
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Time per Token (ms)', fontsize=11)
        ax.set_ylim(0, total_ms * 1.2)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9)
        
        # Add note about H2D transfer
        if h2d_ms > 0:
            ax.text(0.5, 0.95, f'Note: KV H2D transfer = {h2d_ms:.2f}ms (one-time)',
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=8, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n{colored(f'Decode breakdown plot saved to {output_file}', 'green')}")
    plt.close()


def print_comparison(results_list):
    """Print comparison table."""
    print("\n" + "="*80)
    print(colored("BENCHMARK COMPARISON", 'cyan', attrs=['bold']))
    print("="*80)

    # Group by prompt length
    from collections import defaultdict
    by_prompt = defaultdict(list)
    for r in results_list:
        by_prompt[r['prompt_length']].append(r)

    for prompt_len in sorted(by_prompt.keys()):
        results = by_prompt[prompt_len]
        print(f"\n{colored(f'Prompt Length: {prompt_len} tokens', 'yellow', attrs=['bold'])}")
        print("-" * 80)

        # Find full attention baseline
        baseline = next((r for r in results if r['attn_mode'] == 'full'), None)

        for r in results:
            mode = r['attn_mode'].upper()

            # Prefill
            prefill_speedup = ""
            if baseline and r['attn_mode'] != 'full':
                speedup = r['prefill_tokens_per_sec'] / baseline['prefill_tokens_per_sec']
                prefill_speedup = f" ({speedup:.2f}x)"

            # Decode
            decode_speedup = ""
            if baseline and r['attn_mode'] != 'full':
                speedup = r['decode_tokens_per_sec'] / baseline['decode_tokens_per_sec']
                decode_speedup = f" ({colored(f'{speedup:.2f}x', 'green' if speedup > 1 else 'red')})"

            # Total
            total_speedup = ""
            if baseline and r['attn_mode'] != 'full':
                speedup = baseline['total_time'] / r['total_time']
                total_speedup = f" ({colored(f'{speedup:.2f}x faster', 'green' if speedup > 1 else 'red')})"

            print(f"{colored(mode, 'cyan'):20s}")
            print(f"  Prefill:       {r['prefill_tokens_per_sec']:8.2f} tok/s {prefill_speedup}")
            print(f"  Decode:        {r['decode_tokens_per_sec']:8.2f} tok/s {decode_speedup}")
            print(f"  Decode Latency: {r['decode_time_avg']*1000:7.2f} ms/tok")
            print(f"  Total Time:    {r['total_time']:8.4f} s {total_speedup}")
            print(f"  Memory:        {r['memory_allocated_gb']:8.2f} GB")
            print()


def main():
    print(colored("="*80, 'cyan'))
    print(colored("KeySifter vs Full Attention - Runtime Efficiency Benchmark", 'cyan', attrs=['bold']))
    print(colored("="*80, 'cyan'))

    # Path to trained KeySifter weights
    weights_path = '/home/afa55/Projects/xKV/xKV/Llama_31_8bi_GQA_dDash16.pt'

    # Test configurations
    configs = [
        # (prompt_length, gen_length)
        # (512, 32),      # Short context
        # (2048, 32),     # Medium context
        (32768, 1024),     # Medium context
        # (65536, 2048),     # Long context
        # (131072, 2048),     # Very long context
    ]

    all_results = []

    for prompt_len, gen_len in configs:
        print(f"\n{colored(f'>>> Testing with {prompt_len} token prompt, {gen_len} token generation', 'yellow', attrs=['bold'])}")

        # Test Full Attention
        try:
            result = benchmark_model('full', prompt_len, gen_len)
            all_results.append(result)
        except Exception as e:
            print(f"Full attention failed: {e}")

        # Wait a bit between tests
        time.sleep(2)

        # Test KeySifter with trained weights
        try:
            result = benchmark_model('keysifter', prompt_len, gen_len, sparse_budget=1024, predictor_path=weights_path)
            all_results.append(result)
        except Exception as e:
            print(f"KeySifter failed: {e}")

        # Test Oracle (Random)
        try:
            result = benchmark_model('oracle', prompt_len, gen_len, sparse_budget=1024, oracle_random_indices=True)
            result['attn_mode'] = 'oracle_random' # Rename for clarity in output
            all_results.append(result)
        except Exception as e:
            print(f"Oracle (Random) failed: {e}")

        # Wait
        time.sleep(2)

        # Test Oracle (Contiguous)
        try:
            result = benchmark_model('oracle', prompt_len, gen_len, sparse_budget=1024, oracle_random_indices=False)
            result['attn_mode'] = 'oracle_contiguous' # Rename for clarity in output
            all_results.append(result)
        except Exception as e:
            print(f"Oracle (Contiguous) failed: {e}")

        # Wait between configurations
        time.sleep(2)

    # Print comparison
    print_comparison(all_results)

    # Plot decode breakdown
    plot_decode_breakdown(all_results)

    # Save results
    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{colored('Results saved to benchmark_results.json', 'green')}")


if __name__ == '__main__':
    main()
