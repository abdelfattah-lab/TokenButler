#!/usr/bin/env python3
"""
Benchmark KeySifter Grid Search
Iterates through various configurations and logs decode throughput to a CSV.
"""

import torch
import time
import sys
import gc
import csv
import os
from tqdm import tqdm
import itertools

# Adjust path to include project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import Llama
from termcolor import colored


PROMPT_LENGTH = 2**19 # 512K
GEN_LENGTH = 2048
SPARSE_BUDGET = 8192
DDASH_VALUES = [4, 6, 8, 16]
INTDIM_VALUES = [64, 128, 256, 512, 1024]
PROD_FREQ_VALUES = [1, 2, 4, 8]

def benchmark_config(attn_mode, prompt_length, gen_length, 
                     sparse_budget=1024, 
                     dDash=16, 
                     intdim=1024, 
                     producer_frequency=4,
                     predictor_path=''):
    """Benchmark a single configuration."""
    
    # Initialize model
    model_kwargs = {
        'model_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'batch_size': 1,
        'max_length': PROMPT_LENGTH + GEN_LENGTH + 128,
        'device': 'cuda:0',
        'dtype': torch.bfloat16,
        'attn_mode': attn_mode,
    }

    if attn_mode == 'keysifter':
        model_kwargs.update({
            'sparse_budget': sparse_budget,
            'chunk_size': 8,
            'rank': 160,
            'dDash': dDash,
            'producer_frequency': producer_frequency,
            'keysifter_intermediate_dim': intdim,
            'predictor_path': predictor_path,
        })
    elif attn_mode == 'shadowkv':
        model_kwargs.update({
            'sparse_budget': sparse_budget,
            'chunk_size': 8,
            'rank': 160,
        })

    # Catch potential OOM or initialization errors
    try:
        llm = Llama(**model_kwargs)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None

    # Create prompt
    base_text = "The quick brown fox jumps over the lazy dog. "
    # Estimate repetitions needed
    repetitions = (prompt_length // 10) + 20 
    text = base_text * repetitions
    input_ids = llm.encode(text)

    # Truncate to exact length
    if input_ids.shape[1] > prompt_length:
        input_ids = input_ids[:, :prompt_length]

    actual_prompt_len = input_ids.shape[1]
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # --- Prefill ---
    # We run prefill but we are mostly interested in decode throughput for this specific request
    # However, prefill is necessary to reach the decode state.
    try:
        with torch.no_grad():
            logits = llm.prefill(input_ids)
    except RuntimeError as e:
        print(f"Error during prefill: {e}")
        del llm
        torch.cuda.empty_cache()
        return None
        
    torch.cuda.synchronize()

    # --- Decode ---
    # Measure wall-clock time for the generation loop
    # We disable per-step profiling to avoid overhead
    
    start_time = time.perf_counter()
    
    # Ensure KV cache is on device (Full Info requires manual transfer from CPU)
    # This is critical for Full Attention baseline performance
    try:
        llm.kv_cache.H2D()
    except Exception:
        pass
    torch.cuda.synchronize()

    # measure per-step time with synchronization for accuracy
    decode_times = []
    
    try:
        with torch.no_grad():
            for i in range(gen_length):
                torch.cuda.synchronize()
                start_step = time.perf_counter()
                
                next_token = logits.argmax(dim=-1)
                position_ids = llm.get_ctx(next_token)
                logits = llm.inference(input_ids=next_token, position_ids=position_ids)
                
                torch.cuda.synchronize()
                step_time = time.perf_counter() - start_step
                decode_times.append(step_time)
        
        avg_decode_time = sum(decode_times) / len(decode_times)
        decode_throughput = 1.0 / avg_decode_time
        
        # Memory usage
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        
    except Exception as e:
        print(f"Error during decode: {e}")
        decode_throughput = 0.0
        memory_allocated = 0.0
    
    # Cleanup
    del llm
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'decode_throughput': decode_throughput,
        'memory_allocated_gb': memory_allocated,
        'actual_prompt_len': actual_prompt_len
    }

def main():
    # Configuration
    # PROMPT_LENGTH = 65536 # 64K
    
    # Path to weights - assuming same as in benchmark_keysifter.py
    # If not found, user might need to adjust or we can try to locate it.
    WEIGHTS_PATH = ''
    if not os.path.exists(WEIGHTS_PATH):
        print(colored(f"Warning: Predictor weights not found at {WEIGHTS_PATH}", "red"))
        # You might want to exit or continue if testing baseline only, 
        # but for keysifter it will likely fail or use random initialization if allowed.
    
    OUTPUT_FILE = 'benchmark_keysifter_grid.csv'
    
    # Grid parameters
    dDash_values = DDASH_VALUES
    intdim_values = INTDIM_VALUES
    prod_freq_values = PROD_FREQ_VALUES
    
    # Generate grid configurations
    grid_configs = list(itertools.product(dDash_values, intdim_values, prod_freq_values))
    
    # Add Baseline (Full Attention) as the FIRST entry
    # Represents: attn_mode='full', and ignored keysifter params
    configs_to_run = [
        {'type': 'baseline'}
    ]
    
    for d, i, p in grid_configs:
        configs_to_run.append({
            'type': 'keysifter',
            'dDash': d,
            'intdim': i,
            'producer_frequency': p
        })
        
    print(f"Total configurations to run: {len(configs_to_run)}")
    
    # Prepare CSV
    fieldnames = ['attn_mode', 'dDash', 'intdim', 'producer_frequency', 'decode_throughput', 'memory_allocated_gb', 'prompt_length']
    
    # Check if file exists to maybe append vs overwrite? 
    # User said "create a script that would write a CSV", implying new run. Overwrite is safer.
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Progress bar
    pbar = tqdm(configs_to_run, desc="Benchmarking")
    
    for config in pbar:
        if config['type'] == 'baseline':
            attn_mode = 'full'
            dDash = 0
            intdim = 0
            prod_freq = 0
            desc = "Baseline (Full)"
        else:
            attn_mode = 'keysifter'
            dDash = config['dDash']
            intdim = config['intdim']
            prod_freq = config['producer_frequency']
            desc = f"KS (d={dDash}, i={intdim}, p={prod_freq})"
            
        pbar.set_description(f"Running {desc}")
        
        # Run benchmark
        result = benchmark_config(
            attn_mode=attn_mode,
            prompt_length=PROMPT_LENGTH,
            gen_length=GEN_LENGTH,
            sparse_budget=SPARSE_BUDGET,
            dDash=dDash,
            intdim=intdim,
            producer_frequency=prod_freq,
            predictor_path=WEIGHTS_PATH
        )
        
        if result:
            row = {
                'attn_mode': attn_mode,
                'dDash': dDash if attn_mode == 'keysifter' else '',
                'intdim': intdim if attn_mode == 'keysifter' else '',
                'producer_frequency': prod_freq if attn_mode == 'keysifter' else '',
                'decode_throughput': f"{result['decode_throughput']:.2f}",
                'memory_allocated_gb': f"{result['memory_allocated_gb']:.2f}",
                'prompt_length': result['actual_prompt_len']
            }
            
            # Write immediately to CSV
            with open(OUTPUT_FILE, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row)
        else:
            print(f"\nFailed config: {desc}")

    print(f"\nBenchmark complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
