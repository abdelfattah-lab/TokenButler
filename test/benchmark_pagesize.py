#!/usr/bin/env python3
"""
Benchmark Page Size vs Decode Latency for Oracle Paged Selection.
"""

import torch
import time
import sys
import gc
import matplotlib.pyplot as plt
import numpy as np
import json
import os

sys.path.insert(0, '/home/afa55/Projects/xKV/xKV')

from models import Llama
from termcolor import colored

def benchmark_page_size(page_size, prompt_length=8192, gen_length=1024, sparse_budget=2048):
    """Benchmark a single page size configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: Page Size = {page_size} | Prompt: {prompt_length} | Budget: {sparse_budget}")
    print(f"{'='*60}")

    # Initialize model
    model_kwargs = {
        'model_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'batch_size': 1,
        'max_length': prompt_length + gen_length + 128, # Enough for prompt + gen
        'device': 'cuda:0',
        'dtype': torch.bfloat16,
        'attn_mode': 'oracle',
        'sparse_budget': sparse_budget,
        'chunk_size': 8,
        'oracle_random_indices': True,
        'page_size': page_size,
    }

    print("Initializing model...")
    try:
        llm = Llama(**model_kwargs)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return None

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

    # Prefill (we don't measure this strictly but needed for context)
    print("Prefilling...")
    llm.prefill(input_ids)

    # Benchmark decode
    print("Decoding...")
    decode_times = []
    
    # Warmup decoding a few steps
    logits = llm.inference(input_ids=torch.tensor([[100]], device='cuda:0'), position_ids=llm.get_ctx(torch.tensor([[100]], device='cuda:0')))

    # Actual measurement
    # We will run gen_length steps
    # To reduce overhead, we won't sample tokens, just pass a dummy token
    next_token = torch.tensor([[1234]], device='cuda:0')

    for i in range(gen_length):
        torch.cuda.synchronize()
        start = time.perf_counter()

        position_ids = llm.get_ctx(next_token)
        logits = llm.inference(input_ids=next_token, position_ids=position_ids)
        
        torch.cuda.synchronize()
        step_time = time.perf_counter() - start
        decode_times.append(step_time)

    avg_decode_time = sum(decode_times) / len(decode_times)
    decode_latency_ms = avg_decode_time * 1000
    
    print(f"  Avg Decode Latency: {decode_latency_ms:.2f} ms/token")
    
    # Cleanup
    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'page_size': page_size,
        'decode_latency_ms': decode_latency_ms,
        'prompt_length': actual_prompt_len,
        'gen_length': gen_length,
        'sparse_budget': sparse_budget
    }

def plot_results(results, output_file='page_size_latency.png'):
    """Plot Page Size vs Decode Latency."""
    page_sizes = [r['page_size'] for r in results]
    latencies = [r['decode_latency_ms'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(page_sizes, latencies, marker='o', linestyle='-', linewidth=2, color='#4ECDC4')
    
    plt.xscale('log', base=2)
    plt.xlabel('Page Size (tokens)', fontsize=12)
    plt.ylabel('Decode Latency per Token (ms)', fontsize=12)
    plt.title('Decode Latency vs Page Size (Oracle Selection)', fontsize=14, fontweight='bold')
    
    # Add labels
    for x, y in zip(page_sizes, latencies):
        plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")


def main():
    print(colored("="*80, 'cyan'))
    print(colored("Page Size vs Decode Latency Benchmark", 'cyan', attrs=['bold']))
    print(colored("="*80, 'cyan'))

    page_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    results = []

    for ps in page_sizes:
        res = benchmark_page_size(ps, prompt_length=8192*2, gen_length=1024, sparse_budget=2048)
        if res:
            results.append(res)
        time.sleep(1)

    # Save data
    with open('page_size_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to page_size_benchmark.json")

    # Plot
    plot_results(results)

if __name__ == '__main__':
    main()
