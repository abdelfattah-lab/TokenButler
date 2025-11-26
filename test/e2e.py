################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
import gc
import json
import numpy as np
from datetime import datetime
from termcolor import colored
from argparse import ArgumentParser, Namespace
import torch.cuda.nvtx as nvtx
from transformers import AutoTokenizer

from data.dataset import Dataset
os.chdir(root_dir)

from models import choose_model_class

dataset_name = "ruler/qa_2"

configs = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "60k": {
            "sparse_budget": 1024,
            "min_prompt_len": 1024*60,
            "baseline_bsz": 8,
            "shadowkv_bsz": 48,
        },
        "122k": {
            "sparse_budget": 2048,
            "min_prompt_len": 1024*122,
            "baseline_bsz": 4,
            "shadowkv_bsz": 24,
        },
        "244k": {
            "sparse_budget": 4096,
            "min_prompt_len": 1024*244,
            "baseline_bsz": 2,
            "shadowkv_bsz": 12,
        }
    }
}


def clear_cuda_state():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def cleanup_llm(llm):
    """Properly cleanup LLM and KV cache"""
    try:
        if hasattr(llm, 'kv_cache'):
            del llm.kv_cache
    except:
        pass
    del llm
    clear_cuda_state()


def save_results(args, results, output_dir="./logs/results"):
    """Save experiment results to JSON file with structured naming"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine method name
    if args.baseline:
        method = "baseline"
    elif args.shadowkv:
        method = f"shadowkv_k{args.rank_k}"
    elif args.xkey:
        method = f"xkey_k{args.rank_k}_v{args.rank_v}_g{args.group_size}"
    elif args.xkv:
        method = f"xkv_k{args.rank_k}_v{args.rank_v}_g{args.group_size}"
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{method}_len{args.prompt_len}_bs{args.bsz}_budget{args.budget}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare data structure
    data = {
        "timestamp": timestamp,
        "args": vars(args),
        "results": results
    }
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(colored(f"\nResults saved to: {filepath}", 'cyan'))
    return filepath


def run_experiment_rounds(llm_class, model_name, args, input_ids, method_name, attn_mode_config):
    """Run multiple rounds of experiments and compute statistics"""
    
    throughputs = []
    peak_memories = []
    
    for round_idx in range(args.num_rounds):
        print(colored(f"\n--- Round {round_idx + 1}/{args.num_rounds} ---", 'yellow'))
        
        try:
            # Initialize model
            llm = llm_class(**attn_mode_config)
            
            # Clear CUDA state before benchmark
            clear_cuda_state()
            
            # Run generation
            _, throughput = llm.batch_generate(
                input_ids.to(llm.device), 
                gen_len=args.gen_len, 
                benchmark=True, 
                temperature=0.6,
                enable_profiler=args.enable_profiler if round_idx == 0 else False,  # Only profile first round
                profiler_output_dir=args.profiler_output_dir,
                profiler_wait_steps=args.profiler_wait_steps, 
                profiler_warmup_steps=args.profiler_warmup_steps,
                profiler_active_steps=args.profiler_active_steps
            )
            
            torch.cuda.synchronize()
            peak_gb = torch.cuda.max_memory_allocated(llm.device) / (1024**3)
            
            throughputs.append(throughput)
            peak_memories.append(peak_gb)
            
            print(f"Round {round_idx + 1}: Throughput = {throughput:.2f} token/s, Peak Memory = {peak_gb:.2f} GB")
            
            # Cleanup
            cleanup_llm(llm)
            
        except Exception as e:
            print(colored(f"Round {round_idx + 1} failed: {e}", 'red'))
            # Still cleanup if possible
            try:
                cleanup_llm(llm)
            except:
                clear_cuda_state()
            raise e
    
    # Compute statistics
    results = {
        "method": method_name,
        "num_rounds": args.num_rounds,
        "throughput": {
            "values": throughputs,
            "mean": float(np.mean(throughputs)),
            "std": float(np.std(throughputs)),
            "min": float(np.min(throughputs)),
            "max": float(np.max(throughputs))
        },
        "peak_memory_gb": {
            "values": peak_memories,
            "mean": float(np.mean(peak_memories)),
            "std": float(np.std(peak_memories)),
            "min": float(np.min(peak_memories)),
            "max": float(np.max(peak_memories))
        }
    }
    
    # Print summary
    print(colored(f"\n{'='*80}", 'green'))
    print(colored(f"Summary for {method_name}", 'green', attrs=['bold']))
    print(colored(f"{'='*80}", 'green'))
    print(f"Throughput: {results['throughput']['mean']:.2f} ± {results['throughput']['std']:.2f} token/s "
          f"(min: {results['throughput']['min']:.2f}, max: {results['throughput']['max']:.2f})")
    print(f"Peak Memory: {results['peak_memory_gb']['mean']:.2f} ± {results['peak_memory_gb']['std']:.2f} GB "
          f"(min: {results['peak_memory_gb']['min']:.2f}, max: {results['peak_memory_gb']['max']:.2f})")
    print(colored(f"{'='*80}\n", 'green'))
    
    return results


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", choices=["meta-llama/Llama-3.1-8B-Instruct"])
    p.add_argument("--prompt_len", type=int, default=65536, help="The prompt length to test. Maximum is 256k.")
    p.add_argument("--bsz", type=int, default=1, help="Override the batch size in configs.")
    p.add_argument("--budget", type=int, default=2048, help="Override the sparse budget in configs.")
    p.add_argument("--gen_len", type=int, default=100, help="The length of the generation.")
    p.add_argument("--profile_layers", type=int, default=None, help="Only profile the first N layers.")
    
    # Experiment configuration
    p.add_argument("--num_rounds", type=int, default=1, help="Number of rounds to run for averaging results.")
    p.add_argument("--output_dir", type=str, default="./logs/results", help="Directory to save JSON results.")
    
    # Profiler arguments
    p.add_argument("--enable_profiler", action='store_true', default=False, help="Enable torch profiler to trace the inference workload.")
    p.add_argument("--profiler_output_dir", type=str, default="./profiler_logs", help="Directory to save profiler traces.")
    p.add_argument("--profiler_wait_steps", type=int, default=2, help="Number of steps to wait before profiling.")
    p.add_argument("--profiler_warmup_steps", type=int, default=2, help="Number of warmup steps before active profiling.")
    p.add_argument("--profiler_active_steps", type=int, default=6, help="Number of steps to actively profile.")
    
    # Method selection
    p.add_argument("--baseline", action='store_true', default=False, help="Evaluate baseline.")
    p.add_argument("--shadowkv", action='store_true', default=False, help="Evaluate ShadowKV.")
    p.add_argument("--xkey", action='store_true', default=False, help="Evaluate xKey.")
    p.add_argument("--xkv", action='store_true', default=False, help="Evaluate xKV.")
    
    # Compression parameters (shared across methods)
    p.add_argument("--rank_k", type=int, default=64, help="Rank for key compression (used by ShadowKV, xKey, xKV).")
    p.add_argument("--rank_v", type=int, default=96, help="Rank for value compression (used by xKey, xKV).")
    p.add_argument("--group_size", type=int, default=1, help="Group size for xKV/xKey.")
    
    # minference
    # FIXME(max410011): Have bug
    p.add_argument("--minference", type=bool, default=False, help="Use minference to accelerate prefilling.")

    return p.parse_args()

if __name__ == '__main__':

    args = parse_args()
    
    # Validate that only one setup is selected
    selected_methods = sum([args.baseline, args.shadowkv, args.xkey, args.xkv])
    if selected_methods == 0:
        raise ValueError("Please select one method to test: --baseline, --shadowkv, --xkey, or --xkv")
    elif selected_methods > 1:
        raise ValueError("Only one method can be tested at a time. Please select only one of: --baseline, --shadowkv, --xkey, or --xkv")

    model_name = args.model_name
    bsz = args.bsz
    prompt_len = args.prompt_len
    sparse_budget = args.budget
    temperature = 0.6
    
    # Print setup configuration
    print(colored("=" * 80, 'cyan'))
    print(colored("Benchmark Configuration", 'cyan', attrs=['bold']))
    print(colored("=" * 80, 'cyan'))
    print(f"Model: {model_name}")
    print(f"Prompt Length: {prompt_len}")
    print(f"Batch Size: {bsz}")
    print(f"Sparse Budget: {sparse_budget}")
    print(f"Generation Length: {args.gen_len}")
    print(f"Number of Rounds: {args.num_rounds}")
    print(f"Output Directory: {args.output_dir}")
    if args.enable_profiler:
        print(colored(f"Profiler: ENABLED (first round only)", 'yellow', attrs=['bold']))
        print(f"  - Output Directory: {args.profiler_output_dir}")
        print(f"  - Wait Steps: {args.profiler_wait_steps}")
        print(f"  - Warmup Steps: {args.profiler_warmup_steps}")
        print(f"  - Active Steps: {args.profiler_active_steps}")
    if args.baseline:
        print(f"Method: Baseline (Full Attention)")
    elif args.shadowkv:
        print(f"Method: ShadowKV (rank_k={args.rank_k})")
    elif args.xkey:
        print(f"Method: xKey (rank_k={args.rank_k}, rank_v={args.rank_v}, group_size={args.group_size})")
    elif args.xkv:
        print(f"Method: xKV (rank_k={args.rank_k}, rank_v={args.rank_v}, group_size={args.group_size})")
    print(colored("=" * 80, 'cyan'))
    print()
    
    # Profile first N layers
    profile_layers = args.profile_layers
    if profile_layers is not None:
        print(colored(f"Only profile the first {profile_layers} layers.", 'yellow'))

    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = Dataset(dataset_name, tokenizer, 256*1024, 100)
    input_ids = torch.cat([dataset[i][0][:, :prompt_len] for i in range(bsz)], dim=0)
    assert input_ids.shape[-1] == prompt_len


    LLM = choose_model_class(model_name)
    
    if args.baseline:
        try:
            ##################### Baseline #####################
            method_name = f"Baseline (prompt_len={prompt_len}, bsz={bsz})"
            attn_config = {
                'model_name': model_name,
                'device': 'cuda:0',
                'batch_size': bsz,
                'max_length': prompt_len,
                'minference': args.minference,
                'profile_layers': profile_layers,
                'attn_mode': 'full',
                'sparse_budget': sparse_budget
            }
            
            results = run_experiment_rounds(LLM, model_name, args, input_ids, method_name, attn_config)
            save_results(args, results, args.output_dir)
            
        except Exception as e:
            print(colored(f"Baseline failed for prompt_len={prompt_len}, bsz={bsz}", 'red'))
            print(e)

    if args.shadowkv:
        try:
            ##################### ShadowKV #####################
            method_name = f"ShadowKV (prompt_len={prompt_len}, bsz={bsz}, rank_k={args.rank_k})"
            attn_config = {
                'model_name': model_name,
                'device': 'cuda:0',
                'batch_size': bsz,
                'max_length': prompt_len,
                'minference': args.minference,
                'profile_layers': profile_layers,
                'attn_mode': 'shadowkv_cpu',
                'rank': args.rank_k,
                'sparse_budget': sparse_budget
            }
            
            results = run_experiment_rounds(LLM, model_name, args, input_ids, method_name, attn_config)
            save_results(args, results, args.output_dir)
            
        except Exception as e:
            print(colored(f"ShadowKV failed for prompt_len={prompt_len}, bsz={bsz}, rank_k={args.rank_k}", 'red'))
            print(e)


    if args.xkey:
        try:
            ##################### xKey #####################
            method_name = f"xKey (prompt_len={prompt_len}, bsz={bsz}, rank_k={args.rank_k}, rank_v={args.rank_v}, group_size={args.group_size})"
            attn_config = {
                'model_name': model_name,
                'device': 'cuda:0',
                'batch_size': bsz,
                'max_length': prompt_len,
                'minference': args.minference,
                'profile_layers': profile_layers,
                'attn_mode': 'shadowkv_xkey_cpu',
                'sparse_budget': sparse_budget,
                'rank_k': args.rank_k,
                'rank_v': args.rank_v,
                'group_size': args.group_size
            }
            
            results = run_experiment_rounds(LLM, model_name, args, input_ids, method_name, attn_config)
            save_results(args, results, args.output_dir)
            
        except Exception as e:
            print(colored(f"xKey failed for prompt_len={prompt_len}, bsz={bsz}, rank_k={args.rank_k}, rank_v={args.rank_v}, group_size={args.group_size}", 'red'))
            print(e)

    if args.xkv:
        try:
            ##################### xKV #####################
            method_name = f"xKV (prompt_len={prompt_len}, bsz={bsz}, rank_k={args.rank_k}, rank_v={args.rank_v}, group_size={args.group_size})"
            attn_config = {
                'model_name': model_name,
                'device': 'cuda:0',
                'batch_size': bsz,
                'max_length': prompt_len,
                'minference': args.minference,
                'profile_layers': profile_layers,
                'attn_mode': 'shadowkv_xkv_cpu',
                'sparse_budget': sparse_budget,
                'rank_k': args.rank_k,
                'rank_v': args.rank_v,
                'group_size': args.group_size
            }
            
            results = run_experiment_rounds(LLM, model_name, args, input_ids, method_name, attn_config)
            save_results(args, results, args.output_dir)
            
        except Exception as e:
            print(colored(f"xKV failed for prompt_len={prompt_len}, bsz={bsz}, rank_k={args.rank_k}, rank_v={args.rank_v}, group_size={args.group_size}", 'red'))
            print(e)