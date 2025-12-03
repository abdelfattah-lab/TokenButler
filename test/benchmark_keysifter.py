#!/usr/bin/env python3
"""
Benchmark KeySifter vs Full Attention for runtime efficiency.
Tests prefill and decode performance across different sequence lengths.
"""

import torch
import time
import sys
import gc
sys.path.insert(0, '/home/afa55/Projects/xKV/xKV')

from models import Llama
from termcolor import colored

def benchmark_model(attn_mode, prompt_length, gen_length, sparse_budget=512):
    """Benchmark a single configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {attn_mode.upper()} | Prompt: {prompt_length} tokens | Gen: {gen_length} tokens")
    print(f"{'='*60}")

    # Initialize model
    model_kwargs = {
        'model_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'batch_size': 1,
        'max_length': 132072,
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
            # TODO: Are we missing intdim here?
            'predictor_path': '',  # Random weights
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

    # Benchmark decode
    print("\n[DECODE]")
    llm.kv_cache.H2D()

    decode_times = []

    for i in range(gen_length):
        torch.cuda.synchronize()
        start = time.perf_counter()

        next_token = logits.argmax(dim=-1)
        position_ids = llm.get_ctx(next_token)
        logits = llm.inference(input_ids=next_token, position_ids=position_ids)

        torch.cuda.synchronize()
        step_time = time.perf_counter() - start
        decode_times.append(step_time)

    avg_decode_time = sum(decode_times) / len(decode_times)
    decode_tokens_per_sec = 1.0 / avg_decode_time

    print(f"  Avg time per token: {avg_decode_time*1000:.2f}ms")
    print(f"  Throughput: {decode_tokens_per_sec:.2f} tokens/s")
    print(f"  Min/Max: {min(decode_times)*1000:.2f}ms / {max(decode_times)*1000:.2f}ms")

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
    }


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

    # Test configurations
    configs = [
        # (prompt_length, gen_length)
        # (512, 32),      # Short context
        # (2048, 32),     # Medium context
        (65536, 32),     # Long context
        (131072, 32),     # Very long context
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

        # Test KeySifter
        try:
            result = benchmark_model('keysifter', prompt_len, gen_len, sparse_budget=1024)
            all_results.append(result)
        except Exception as e:
            print(f"KeySifter failed: {e}")

        # Wait between configurations
        time.sleep(2)

    # Print comparison
    print_comparison(all_results)

    # Save results
    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{colored('Results saved to benchmark_results.json', 'green')}")


if __name__ == '__main__':
    main()
