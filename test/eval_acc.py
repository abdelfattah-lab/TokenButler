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

# OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 8 test/eval_acc.py --datalen 131072 --method shadowKV --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multikey_3,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/cwe,ruler/fwe,ruler/qa_1,ruler/qa_2" --sparse_budget 896 --rank 160 --chunk_size 8

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import warnings
warnings.filterwarnings("ignore")

import torch
import gc
from termcolor import colored
from argparse import ArgumentParser, Namespace

import torch.distributed as dist
import datetime

import numpy as np
import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

class DistConfig:
    def __init__(self, is_distributed, rank, world_size, device, master_process):
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.master_process = master_process

def init_dist():
    rank = int(os.environ.get("RANK", -1))
    is_distributed = rank != -1
    if is_distributed:
        dist.init_process_group(backend="nccl",timeout=datetime.timedelta(seconds=60*90))
        world_size = int(os.environ["WORLD_SIZE"])

        device = f"cuda:{rank}" 
        torch.cuda.set_device(device)
        master_process = (
            rank == 0
        )
    else:
        device = "cuda:0"
        world_size = 1
        master_process = True

    if master_process:
        print(colored(f"[Dist init] world_size={world_size}", 'cyan'))
    
    return DistConfig(is_distributed, rank, world_size, device, master_process)

def parse_args() -> Namespace:
    def str_to_list(arg):
        return arg.split(',')
    p = ArgumentParser()
    p.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--dataset_name", type=str_to_list, default=["ruler/niah_single_1"])
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--datalen", type=int, default=128*1024, help="The length of the context.")
    p.add_argument("--method", type=str, default="full")
    # mInference args
    p.add_argument("--minference", action='store_true', default=False)
    # ShadowKV args
    p.add_argument("--sparse_budget", type=int, default=2048)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--rank", type=int, default=160)
    # xKV args
    p.add_argument("--group_size", type=int, default=1)
    p.add_argument("--rank_k", type=int, default=96)
    p.add_argument("--rank_v", type=int, default=144)
    p.add_argument("--fake_svd", action='store_true', help="Use fake SVD.")
    # KeySifter args
    p.add_argument("--predictor_path", type=str, default="", help="Path to KeySifter predictor weights (.pt file)")
    p.add_argument("--dDash", type=int, default=32, help="Reduced dimension for KeySifter importance computation")
    p.add_argument("--producer_frequency", type=int, default=4, help="Number of layers served by one KeySifter predictor")
    p.add_argument("--keysifter_intermediate_dim", type=int, default=512, help="KeySifter MLP intermediate dimension")
    p.add_argument("--predict_interval", type=int, default=1, help="Predict important tokens every N decode tokens (1=every token, baseline)")
    p.add_argument("--enable_neighbor_fetch", action='store_true', default=False, help="Enable neighbor fetching with 2x sparse buffer")
    p.add_argument("--force_query_prediction", action='store_true', default=False, help="Force prediction at the start of each new query in multi-turn (refreshes sparse selection immediately)")
    p.add_argument("--no_prefill_cont_dense", action='store_true', default=False, help="Disable dense prediction during prefill_cont (use config's predict_interval instead of forcing i=1)")
    p.add_argument("--inference_mode", type=str, default="single_turn", choices=["single_turn", "multi_turn"],
                   help="Inference mode: single_turn (combined prompt) or multi_turn (prefill context, then query)")
    p.add_argument("--sparse_turns", type=int, default=-1,
                   help="Number of turns to use sparse attention (-1=all sparse). "
                        "After this many turns, switch to dense attention for remaining turns.")
    p.add_argument("--output_dir", type=str, default="",
                   help="Override output directory for result files. "
                        "If set, results are written here instead of the default archive/ path.")

    return p.parse_args()

if __name__ == '__main__':

    args = parse_args()
    model_name = args.model_name
    batch_size = args.batch_size
    dataset_names = args.dataset_name
    num_samples = args.num_samples
    datalen = args.datalen
    sparse_budget = args.sparse_budget
    dtype = torch.bfloat16
    rank = args.rank
    chunk_size = args.chunk_size
    minference = args.minference
    group_size = args.group_size
    rank_k = args.rank_k
    rank_v = args.rank_v
    fake_svd = args.fake_svd
    predictor_path = args.predictor_path
    dDash = args.dDash
    producer_frequency = args.producer_frequency
    keysifter_intermediate_dim = args.keysifter_intermediate_dim

    seed_everything(42)
    dist_config = init_dist()
    
    from evaluator import Evaluator
    from models import choose_model_class
    from data.dataset import Dataset
    
    evaluator = Evaluator(dist_config)
    
    if dist_config.master_process:
        print(colored(f"data_names: {dataset_names}", 'cyan'))

    LLM = choose_model_class(model_name)

    llm = LLM(model_name=model_name, batch_size=batch_size, device=dist_config.device, max_length=datalen+2048, attn_mode=args.method, dtype=dtype,
              sparse_budget=sparse_budget, chunk_size=chunk_size, rank=rank, minference=minference, rank_k=rank_k, rank_v=rank_v, group_size=group_size, fake_svd=fake_svd,
              predictor_path=predictor_path, dDash=dDash, producer_frequency=producer_frequency, keysifter_intermediate_dim=keysifter_intermediate_dim,
              predict_interval=args.predict_interval, enable_neighbor_fetch=args.enable_neighbor_fetch)

    # Set prefill_cont_dense on the kv_cache
    if args.no_prefill_cont_dense and hasattr(llm.kv_cache, 'prefill_cont_dense'):
        llm.kv_cache.prefill_cont_dense = False

    if dist_config.master_process:
        llm.print_kv_stats()

    for dataset_name in dataset_names:
        dataset = Dataset(dataset_name, llm.tokenizer, datalen, num_samples, evaluator.dist_config.rank, evaluator.dist_config.world_size, inference_mode=args.inference_mode)
        result_fname = f"{dataset_name}_{datalen}_{args.method}_b{sparse_budget}_c{chunk_size}_x{args.group_size}_r{rank}_k{rank_k}_v{rank_v}.jsonl"
        if args.output_dir:
            # Strip dataset subdirectory prefix (e.g. "ruler/") for flat output
            result_fname = os.path.basename(result_fname)
            result_path = os.path.join(args.output_dir, result_fname)
        else:
            result_path = f"archive/{model_name.split('/')[-1]}/{result_fname}"
        evaluator.test(llm, dataset, result_path, args.method, sparse_turns=args.sparse_turns)

    evaluator.summarize()