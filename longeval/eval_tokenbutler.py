# evaluate/eval_tokbutler.py

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import gc
from termcolor import colored
import argparse
from argparse import ArgumentParser, Namespace

import torch.distributed as dist
import datetime
import json

import numpy as np
import random

# Make sure repo root is on sys.path (same as eval_acc.py)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# =============================================================================
# Distributed helpers (copied from your eval_acc.py)
# =============================================================================

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
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=60 * 90),
            device_id=torch.device(device),
        )
        master_process = rank == 0
    else:
        device = "cuda:0"
        world_size = 1
        master_process = True

    if master_process:
        print(colored(f"[Dist init] world_size={world_size}", "cyan"))

    return DistConfig(is_distributed, rank, world_size, device, master_process)

# =============================================================================
# TokenButler / AttentionExperimental helpers (adapted from your generation script)
# =============================================================================

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from predictor import PredictorDynamicCache  # your predictor cache

def maybe_hf_login():
    token = os.getenv("HFTOKEN")
    if token:
        try:
            login(token=token)
        except Exception as e:
            print(f"Warning: HF login failed ({e}). Continuing without login.")

def set_inference_mode(model, mode: bool):
    """
    Sets the `inference_mode` flag for all *AttentionExperimental modules.
    """
    for module in model.modules():
        if module.__class__.__name__.endswith("AttentionExperimental"):
            module.inference_mode = mode

def patched_prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
    """
    Patch HF generation: inject PredictorDynamicCache instead of DynamicCache.
    """
    if "past_key_values" not in model_kwargs or model_kwargs["past_key_values"] is None:
        model_kwargs["past_key_values"] = PredictorDynamicCache()
    return model_kwargs

def infer_producer_frequency(config):
    """
    Infer how many layers the model has, used as producer_frequency.
    """
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    if hasattr(config, "num_layers"):
        return config.num_layers
    raise ValueError("Could not infer number of layers from config (no num_hidden_layers/num_layers).")

def get_producer_layers(model):
    """
    Return all attention blocks that *own* a predictor, i.e. the group roots.
    We identify them via `is_predictor_owner`, and sort by `layer_idx` so the
    order matches how the checkpoint list was saved.
    """
    from modify_models.modify_llama import LlamaAttentionExperimental  # adjust import if needed

    producers = []
    for m in model.modules():
        if isinstance(m, LlamaAttentionExperimental) and getattr(m, "is_predictor_owner", False):
            producers.append(m)

    # Make sure ordering is stable / matches training time
    producers.sort(key=lambda m: m.layer_idx)
    return producers
def replace_attention_modules(model, config, args):
    """
    Replace standard attention with *AttentionExperimental via convert_kvcache_experimental
    for the chosen architecture.
    """
    model_path = args.model_name_or_path

    if args.architecture == "llama" and "Yarn-Llama" not in model_path:
        print("Running LLaMA module replacement")
        if args.eval_llm_mode in ["ExpPred", "ReplAttn"]:
            from modify_models.modify_llama import (
                convert_kvcache_experimental,
                LlamaAttentionExperimental,  # noqa: F401
            )
        else:
            from modify_models.modify_llama_baselines import (
                convert_kvcache_experimental,
                LlamaAttentionExperimental,  # noqa: F401
            )
        model = convert_kvcache_experimental(model, config, args.producer_frequency)

    elif args.architecture == "mistral":
        print("Running Mistral module replacement")
        if args.eval_llm_mode == "ExpPred":
            from modify_models.modify_mistral import convert_kvcache_experimental
        else:
            from modify_models.modify_mistral_baselines import convert_kvcache_experimental
        model = convert_kvcache_experimental(model, config, args.producer_frequency)

    elif args.architecture == "mixtral":
        print("Running Mixtral module replacement")
        if args.eval_llm_mode == "ExpPred":
            from modify_models.modify_mixtral import convert_kvcache_experimental
        else:
            raise NotImplementedError("Baseline modes not implemented for Mixtral yet.")
        model = convert_kvcache_experimental(model, config, args.producer_frequency)

    elif args.architecture == "phi3":
        print("Running Phi-3 module replacement")
        if args.eval_llm_mode == "ExpPred":
            from modify_models.modify_phi3 import convert_kvcache_experimental
        else:
            from modify_models.modify_phi3_baselines import convert_kvcache_experimental
        model = convert_kvcache_experimental(model, config, args.producer_frequency)

    elif args.architecture == "glm":
        print("Running GLM module replacement")
        if args.eval_llm_mode == "ExpPred":
            from modify_models.modify_glm import convert_kvcache_experimental
        else:
            raise NotImplementedError("Baseline modes not implemented for GLM yet.")
        model = convert_kvcache_experimental(model, config, args.producer_frequency)

    elif args.architecture == "qwen":
        print("Running Qwen module replacement")
        if args.eval_llm_mode == "ExpPred":
            from modify_models.modify_qwen import convert_kvcache_experimental
        else:
            raise NotImplementedError("Baseline modes not implemented for Qwen yet.")
        model = convert_kvcache_experimental(model, config, args.producer_frequency)

    else:
        raise NotImplementedError(f"Architecture {args.architecture} not supported.")

    return model
    
def configure_experimental_modules(model, args):
    """
    Set per-module attributes for *AttentionExperimental modules and
    compute a rough expected token sparsity for logging.
    """
    token_sparsity_list = []

    # Nominal sequence length to estimate sparsity for fixed_ytok
    nominal_seq_len = getattr(args, "datalen", None)
    if nominal_seq_len is None and hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
        nominal_seq_len = model.config.max_position_embeddings
    if nominal_seq_len is None:
        nominal_seq_len = 2048  # safe fallback

    for module in model.modules():
        if module.__class__.__name__.endswith("AttentionExperimental"):
            module.eval_llm_mode = args.eval_llm_mode
            module.token_sparse_method = args.token_sparse_method

            # Let the module parse token_sparse_method and set:
            #   target_sparsity / target_keep_tokens / sparse_aggression / head_keep
            module.set_token_sparsity()

            # ---- estimate sparsity for logging ----
            method = module.token_sparse_method or ""
            expected_sparsity = 0.0

            if "pc" in method:
                # fixed_xpc: target_sparsity is "fraction of candidates pruned"
                if getattr(module, "target_sparsity", None) is not None:
                    expected_sparsity = module.target_sparsity
                elif getattr(module, "sparse_aggression", None) is not None:
                    expected_sparsity = 1.0 - float(module.sparse_aggression)
                else:
                    expected_sparsity = 0.0

            elif "tok" in method:
                # fixed_ytok: approximate sparsity at a nominal sequence length
                keep_tokens = getattr(module, "target_keep_tokens", None)
                if keep_tokens is not None and keep_tokens > 0:
                    head = getattr(args, "min_sparse_index", 0) or 0
                    tail = getattr(args, "sliding_window", 0) or 0
                    eff_context = max(1, nominal_seq_len - head - tail)
                    keep = min(keep_tokens, eff_context)
                    expected_sparsity = 1.0 - (keep / float(eff_context))
                else:
                    # e.g. layer 0 (we keep it dense)
                    expected_sparsity = 0.0
            else:
                # No sparsity / unsupported scheme
                expected_sparsity = 0.0

            token_sparsity_list.append(expected_sparsity)

            # ---- copy the rest of the config into the module ----
            module.stream_llm_start_size = args.stream_llm_start_size
            module.num_tok_per_page = args.num_tok_per_page
            module.producer_frequency = args.producer_frequency
            module.dDash = args.dDash
            module.attn_reduce_factor = args.attn_reduce_factor
            module.head_attn_reduce_factor = args.head_attn_reduce_factor
            module.intdim = args.intdim
            module.flash_attn = args.flash_attn
            module.train_headpredictor = args.train_headpredictor
            module.min_sparse_index = args.min_sparse_index
            module.lookahead = args.lookahead
            module.num_layers_pred = args.producer_frequency
            module.sliding_window = args.sliding_window
            module.tokenbutler_variant = args.tokenbutler_variant

            if args.eval_llm_mode in ["ExpPred", "ReplAttn"]:
                if module.layer_idx % args.producer_frequency == 0:
                    module.update_predictor()

    if token_sparsity_list:
        avg_token_sparsity = sum(token_sparsity_list) / len(token_sparsity_list)
        print(f"Average expected token sparsity: {avg_token_sparsity:.4f}")
        return avg_token_sparsity

    return None

def load_producer_weights_if_needed(model, predictor_ckpt: str):
    """
    Load trained producer / predictor layer weights from a checkpoint or raw list.

    Expects either:
      - a list: [state_dict_layer0, state_dict_layer1, ...]
      - or a dict with key 'model_state_dict' containing that list.
    """
    if predictor_ckpt is None:
        return

    print(f"Loading producer layer weights from: {predictor_ckpt}")
    ckpt = torch.load(predictor_ckpt, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        producer_layer_weights = ckpt["model_state_dict"]
    else:
        producer_layer_weights = ckpt

    model_producer_layers = get_producer_layers(model)

    if len(producer_layer_weights) != len(model_producer_layers):
        print(
            f"Warning: #weights ({len(producer_layer_weights)}) != #producer_layers ({len(model_producer_layers)}). "
            "Will load as many as possible."
        )

    for idx, producer_layer_weight in enumerate(producer_layer_weights):
        if idx >= len(model_producer_layers):
            break
        try:
            model_producer_layers[idx].load_state_dict(producer_layer_weight, strict=False)
        except Exception as e:
            print(f"Error loading producer layer {idx}: {e}")
            print("Continuing; some predictor layers may remain at init weights.")

# =============================================================================
# Arg parsing: reuse RULER args + add TokenButler flags
# =============================================================================

def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--model_name_or_path', type=str, help='model to load')
    parser.add_argument('--flash2', action='store_true', help='whether to use flash-attention2')
    parser.add_argument('--xKV', action='store_true', help='whether to enable xKV patch')
    parser.add_argument('--streamingllm', action='store_true', help='whether to enable StreamingLLM patch')
    parser.add_argument('--snapKV', action='store_true', help='whether to enable snapKV patch')
    parser.add_argument('--pyramidkv', action='store_true', help='whether to enable pyramidkv patch')
    parser.add_argument('--kivi', action='store_true', help='whether to enable KIVI patch')
    parser.add_argument('--quest', action='store_true', help='whether to enable Quest patch')


    # online svd options
    # SVD-related parameters
    parser.add_argument("--rank_k", type=int, default=256, help="Rank for SVD compression of keys")
    parser.add_argument("--rank_v", type=int, default=768, help="Rank for SVD compression of values")
    parser.add_argument(
        '--layer_group_size',
        type=int,
        default=1,
        help='The number of layers that will be grouped and decompose jointly'
    )
    
    parser.add_argument(
        '--layer_merge_impl', 
        type=str, 
        default='svd',
        help='The implementation for layer merge'
    )
    parser.add_argument(
        '--slerp_t',
        type=float,
        default=0.5,
        help='The interpolation ratio for SLERP'
    )
    parser.add_argument(
        '--slerp_gamma',
        type=float,
        default=0.05,
        help='The gamma for identifying divergent token in SLERP',
    )
    
    # Merge control
    parser.add_argument("--merge_key", action="store_true", help="Enable merging for keys")
    parser.add_argument("--merge_value", action="store_true", help="Enable merging for values")
    parser.add_argument("--start_layer_idx", type=int, default=0, help="The starting layer index for layer merging")
    parser.add_argument("--end_layer_idx", type=int, default=-1, help="The ending layer index for layer merging. If -1, it will be the last layer.")
    parser.add_argument('--customized_merge_config', type=str, help='custom config file')
    
    # Quantization parameters
    parser.add_argument('--kv_bits', type=int, default=16, help='KV cache bit width, 16 means no quantization')
    parser.add_argument('--group_size', type=int, default=0, help='Group size for quantization, 0 means per-token quantization')
    parser.add_argument('--sym', action='store_true', help='Use symmetric quantization')
    parser.add_argument('--clip_ratio', type=float, default=1.0, help='Clip ratio for quantization')
    parser.add_argument('--hadamard', action='store_true', help='Use Hadamard transform for quantization')
    
    return parser


def parse_args() -> Namespace:
    def str_to_list(arg):
        return arg.split(',')

    p = ArgumentParser()
    # from utils import add_common_args  # from RULER’s utils.py
    add_common_args(p)

    # RULER-specific flags
    p.add_argument("--dataset_name", type=str_to_list, default=["ruler/niah_single_1"])
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--datalen", type=int, default=65536, help="The length of the context.")
    p.add_argument("--result_dir", type=str, default="results")
    p.add_argument("--use_chat_template", action="store_true", help="Whether to use chat template for long_bench tasks")

    # TokenButler / experimental attention specific flags
    p.add_argument(
        "--architecture",
        type=str,
        default="llama",
        choices=["llama", "mistral", "mixtral", "qwen", "glm", "phi3"],
    )
    p.add_argument(
        "--eval_llm_mode",
        type=str,
        default="ExpPred",
        help="ExpPred, ReplAttn, etc. Used by AttentionExperimental.",
    )
    p.add_argument(
        "--token_sparse_method",
        type=str,
        default="fixed_10pc",
        help="e.g. LazyLLM, progressive_xpc, fixed_xpc, etc.",
    )
    p.add_argument("--stream_llm_start_size", type=int, default=4)
    p.add_argument("--num_tok_per_page", type=int, default=16)
    p.add_argument("--min_sparse_index", type=int, default=4)
    p.add_argument("--attn_reduce_factor", type=int, default=8)
    p.add_argument("--head_attn_reduce_factor", type=int, default=2)
    p.add_argument("--dDash", type=int, default=16)
    p.add_argument("--intdim", type=int, default=512)
    p.add_argument("--lookahead", type=int, default=0)
    p.add_argument("--sliding_window", type=int, default=None)
    p.add_argument("--train_headpredictor", action="store_true")
    p.add_argument("--override_dense", action="store_true")
    p.add_argument("--flash_attn", action="store_true")
    
    p.add_argument('--producer_frequency', type=int, default=None, help='Frequency of predictor')
    p.add_argument(
        '--tokenbutler',
        action='store_true',
        help='Use original TokenButler predictor (baseline: learned Q+K mini-transformer).',
    )
    p.add_argument(
        '--tokenbutler_slice',
        action='store_true',
        help='Use TokenButler variant with learned Q and K taken as the first dDash dims of the real key cache.',
    )
    p.add_argument(
        '--tokenbutler_project',
        action='store_true',
        help='Use TokenButler variant with learned Q and a learned linear projection of the real key cache.',
    )
    p.add_argument(
        "--predictor_ckpt",
        type=str,
        default=None,
        help="Path to checkpoint containing producer/predictor weights (.pt).",
    )

    return p.parse_args()

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name_or_path   # provided by add_common_args(...)
    dataset_names = args.dataset_name
    num_samples = args.num_samples
    datalen = args.datalen

    seed_everything(42)
    dist_config = init_dist()

    if dist_config.master_process:
        os.makedirs("temporary", exist_ok=True)

    from evaluator import Evaluator
    from data.dataset import Dataset

    evaluator = Evaluator(dist_config)

    if dist_config.master_process:
        print(colored(f"data_names: {dataset_names}", "cyan"))

    # ------------------- Build TokenButler model -------------------
    maybe_hf_login()

    # (Optional) flash-sdp if you want, safe to omit
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except Exception:
        pass

    variant_flags = [
        args.tokenbutler,
        args.tokenbutler_slice,
        args.tokenbutler_project,
    ]
    if sum(bool(x) for x in variant_flags) > 1:
        raise ValueError(
            "Please specify at most one of "
            "--tokenbutler, --tokenbutler_slice, or --tokenbutler_project."
        )
    if args.tokenbutler_slice:
        args.tokenbutler_variant = "tokenbutler_slice"
    elif args.tokenbutler_project:
        args.tokenbutler_variant = "tokenbutler_project"
    else:
        # Default: original predictor (paper behaviour)
        args.tokenbutler_variant = "tokenbutler"


    # Load tokenizer & config
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=True,
        use_fast=True,
        trust_remote_code=True,
    )
    config = AutoConfig.from_pretrained(
        model_name,
        use_auth_token=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.producer_frequency is None:
        args.producer_frequency = infer_producer_frequency(config)

    if dist_config.master_process:
        print(colored(f"Loading base model from {model_name}", "cyan"))

    # One copy per rank (same pattern as RULER’s eval)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(dist_config.device)

    model.eval()

    # Replace attention with your *AttentionExperimental
    model = replace_attention_modules(model, config, args)

    # Configure sparsity/predictor parameters
    avg_token_sparsity = configure_experimental_modules(model, args)
    if dist_config.master_process and avg_token_sparsity is not None:
        print(
            colored(
                f"Configured experimental modules with avg token sparsity: {avg_token_sparsity:.4f}",
                "cyan",
            )
        )

    # Load predictor weights if provided
    if args.predictor_ckpt is not None:
        load_producer_weights_if_needed(model, args.predictor_ckpt)

    # Patch HF generation to use PredictorDynamicCache
    model._prepare_cache_for_generation = patched_prepare_cache_for_generation.__get__(
        model,
        model.__class__,
    )

    # Put attention modules into inference mode (ExpPred decode-time sparsity)
    set_inference_mode(model, True)

    if dist_config.master_process:
        print(colored("TokenButler model ready, starting RULER evaluation...", "green"))

    if args.override_dense:
        # One copy per rank (same pattern as RULER’s eval)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(dist_config.device)

        model.eval()

    # ------------------- RULER evaluation loop -------------------
    for dataset_name in dataset_names:
        dataset = Dataset(
            dataset_name,
            tokenizer,
            datalen,
            num_samples,
            evaluator.dist_config.rank,
            evaluator.dist_config.world_size,
            use_chat_template=args.use_chat_template,
        )

        archive_root = os.path.join("temporary", model_name.split('/')[-1])
        os.makedirs(archive_root, exist_ok=True)

        file_name = f"{dataset_name}_{datalen}_tokbutler.jsonl"
        archive_path = os.path.join(archive_root, file_name)
        print(f"\n\nArchive path: {archive_path}\n\n")

        evaluator.test(model, tokenizer, dataset, archive_path)

        stats = evaluator.all_stats[-1]
        benchmark_name = dataset_name.split('/')[-2]
        raw_model_name = model_name.split('/')[-1]

        df = evaluator.summarize()

        if dist_config.master_process:
            df = df.reset_index(drop=True)
            result = df[df["dataset"] == dataset_name]
            per_dataset_stats = result.to_dict(orient="records")[0]
            print(colored(f"Results for {dataset_name}: {per_dataset_stats}", "cyan"))

            # Save JSON log per dataset
            os.makedirs(os.path.join(args.result_dir, f"{benchmark_name}"), exist_ok=True)
            out_json = os.path.join(args.result_dir, f"{benchmark_name}/{raw_model_name}_tokbutler.json")
            with open(out_json, "a") as f:
                meta_data_to_log = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "args": vars(args),
                }
                meta_data_to_log.update(per_dataset_stats)
                json.dump(meta_data_to_log, f)
                f.write("\n")
            # Also log per-example outputs (questions + model generations) into results dir
            # Copy from the temporary archive JSONL created by evaluator.test(...)
            dataset_id = dataset_name.split("/")[-1]
            outputs_jsonl = os.path.join(
                args.result_dir,
                f"{benchmark_name}/{raw_model_name}_{dataset_id}_tokbutler_outputs.jsonl",
            )
            with open(archive_path, "r") as src, open(outputs_jsonl, "a") as dst:
                for line in src:
                    dst.write(line)
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    evaluator.summarize(shown_avg=True)

    if dist_config.is_distributed:
        dist.destroy_process_group()
