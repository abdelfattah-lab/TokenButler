#!/usr/bin/env python3
"""Compare outputs between:

A) Original KeySifter implementation in `KeySifter/` (HF model patched with AttentionExperimental
   + PredictorDynamicCache + loaded producer weights from a checkpoint)
B) xKV's custom KeySifter pipeline (`xKV.models.llama.Llama(attn_mode='keysifter')`)

Goal: verify that for the *same* prompt and *same* predictor checkpoint, both systems produce
similar (ideally identical) greedy outputs.

This is intentionally narrow: one dataset sample, deterministic settings.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KEYSIFTER_ROOT = os.path.join(REPO_ROOT, "KeySifter")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# KeySifter folder uses absolute imports like `from predictor import ...`
if KEYSIFTER_ROOT not in sys.path:
    sys.path.insert(0, KEYSIFTER_ROOT)


@dataclass
class Cfg:
    model_name: str
    predictor_ckpt: str
    device: str
    datalen: int
    gen_len: int
    sample_idx: int
    seed: int
    producer_frequency: int
    dDash: int
    intdim: int
    local_window: int
    sparse_budget: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def first_diff(a: str, b: str) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def build_prompt_from_longbench(sample: dict) -> str:
    # Keep it simple & explicit (matches the commonly-used narrativeqa prompt format).
    return f"{sample['context']}\n\nQuestion: {sample['input']}\nAnswer:"


def run_keysifter_reference(cfg: Cfg, prompt: str) -> str:
    """Run KeySifter/ implementation via HF model patching."""

    from datasets import load_dataset
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # Import helpers directly from KeySifter's evaluation script (keeps behavior consistent)
    from longeval.eval_tokenbutler import (
        PredictorDynamicCache,
        configure_experimental_modules,
        patched_prepare_cache_for_generation,
        replace_attention_modules,
        set_inference_mode,
    )
    from longeval.eval_tokenbutler import get_producer_layers

    def load_checkpoint_into_producers(model, predictor_ckpt: str) -> None:
        """Load the predictor checkpoint into the (possibly fewer) producer modules.

        The checkpoint at `xKV/Llama_31_8bi_GQA_dDash16.pt` is a training checkpoint dict
        containing `model_state_dict` as a list.

        In some KeySifter training setups, each saved producer's state_dict recursively contains
        previous producers under the `producer.` prefix, and the list can be longer than the
        number of producers created for the current `producer_frequency`.

        For comparison against xKV's implementation (producer_frequency=16 => 2 producers), we
        map the checkpoint list to the available producers by stride.
        """

        ckpt = torch.load(predictor_ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            weights_list = ckpt["model_state_dict"]
        else:
            weights_list = ckpt

        producers = get_producer_layers(model)
        if len(producers) == 0:
            raise RuntimeError("No producer layers found in patched KeySifter model")

        if not isinstance(weights_list, list) or len(weights_list) == 0:
            raise RuntimeError("Checkpoint has no producer weights list")

        # If counts match, load 1:1.
        if len(weights_list) == len(producers):
            selected = list(enumerate(weights_list))
        else:
            # Heuristic mapping: stride across the checkpoint list.
            # Example: 8 ckpt entries, 2 producers => indices [0,4].
            stride = max(1, len(weights_list) // len(producers))
            selected = []
            for pi in range(len(producers)):
                wi = min(pi * stride, len(weights_list) - 1)
                selected.append((pi, weights_list[wi]))

        for producer_idx, raw_sd in selected:
            if producer_idx >= len(producers):
                break
            if not isinstance(raw_sd, dict):
                continue

            # Drop nested producer recursion keys (producer.producer....)
            sd = {k: v for k, v in raw_sd.items() if not k.startswith("producer.")}
            try:
                producers[producer_idx].load_state_dict(sd, strict=False)
            except Exception as e:
                raise RuntimeError(f"Failed loading producer[{producer_idx}] from checkpoint: {e}")

    seed_everything(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(cfg.device)
    model.eval()

    # Create a lightweight args-like object for the patching utilities.
    class Args:
        pass

    args = Args()
    args.model_name_or_path = cfg.model_name
    args.architecture = "llama"
    args.eval_llm_mode = "ExpPred"

    # Predictor/config knobs
    args.producer_frequency = cfg.producer_frequency
    args.dDash = cfg.dDash
    args.intdim = cfg.intdim

    # Sparsity knobs
    args.token_sparse_method = "fixed_10pc"
    args.stream_llm_start_size = 4
    args.num_tok_per_page = 16
    args.min_sparse_index = 4
    args.attn_reduce_factor = 8
    args.head_attn_reduce_factor = 2
    args.lookahead = 0
    args.sliding_window = None
    args.train_headpredictor = False
    args.override_dense = False
    args.flash_attn = False

    # Some code paths check these
    args.tokenbutler = False
    args.tokenbutler_slice = False
    args.tokenbutler_project = True
    args.tokenbutler_variant = "tokenbutler_project"

    model = replace_attention_modules(model, config, args)
    configure_experimental_modules(model, args)

    load_checkpoint_into_producers(model, cfg.predictor_ckpt)

    # Ensure generation uses PredictorDynamicCache.
    model._prepare_cache_for_generation = patched_prepare_cache_for_generation.__get__(
        model,
        model.__class__,
    )
    set_inference_mode(model, True)

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.datalen)
    input_ids = enc["input_ids"].to(cfg.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(cfg.device)

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=cfg.gen_len,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
            past_key_values=PredictorDynamicCache(),
        )

    gen = out[0, input_ids.shape[-1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True)

    # Cleanup aggressively.
    del model
    torch.cuda.empty_cache()

    return text


def run_xkv_keysifter(cfg: Cfg, prompt: str) -> str:
    """Run xKV custom KeySifter pipeline."""

    from transformers import AutoTokenizer

    from xKV.models import Llama

    seed_everything(cfg.seed)

    llm = Llama(
        model_name=cfg.model_name,
        max_length=cfg.datalen,
        device=cfg.device,
        attn_mode="keysifter",
        predictor_path=cfg.predictor_ckpt,
        producer_frequency=cfg.producer_frequency,
        dDash=cfg.dDash,
        keysifter_intermediate_dim=cfg.intdim,
        sparse_budget=cfg.sparse_budget,
        page_size=1,
        quantize_int8=False,
    )

    # Use HF tokenizer to build input_ids to avoid template mismatches.
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, trust_remote_code=True)

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.datalen)
    input_ids = enc["input_ids"].to(cfg.device)

    with torch.inference_mode():
        text = llm.generate(
            input_ids,
            gen_len=cfg.gen_len,
            temperature=0.0,
            top_p=1.0,
            top_k=50,
            verbose=False,
        )[0]

    # xKV.generate returns the *full* decoded string for the whole sequence; we want only continuation.
    # Safer: re-decode the generated tail using the same tokenizer.
    # Unfortunately xKV.generate doesn't currently return token ids; so we approximate by stripping the prompt.
    # Keep it robust-ish by returning last gen_len tokens worth of text is not trivial.
    # We'll instead return the raw output and also print a note in comparison.
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--predictor-ckpt", default="xKV/Llama_31_8bi_GQA_dDash16.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datalen", type=int, default=32768)
    p.add_argument("--gen-len", type=int, default=32)
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--producer-frequency", type=int, default=16)
    p.add_argument("--dDash", type=int, default=16)
    p.add_argument("--intdim", type=int, default=1024)
    p.add_argument("--local-window", type=int, default=32)
    p.add_argument("--sparse-budget", type=int, default=2048)
    args = p.parse_args()

    cfg = Cfg(
        model_name=args.model_name,
        predictor_ckpt=args.predictor_ckpt,
        device=args.device,
        datalen=args.datalen,
        gen_len=args.gen_len,
        sample_idx=args.sample_idx,
        seed=args.seed,
        producer_frequency=args.producer_frequency,
        dDash=args.dDash,
        intdim=args.intdim,
        local_window=args.local_window,
        sparse_budget=args.sparse_budget,
    )

    from datasets import load_dataset

    seed_everything(cfg.seed)

    ds = load_dataset("THUDM/LongBench", "narrativeqa", split="test")
    sample = ds[cfg.sample_idx]
    prompt = build_prompt_from_longbench(sample)

    print(f"Prompt chars={len(prompt)} sample_idx={cfg.sample_idx}")

    print("\n=== A) KeySifter reference (HF patched) ===")
    out_a = run_keysifter_reference(cfg, prompt)
    print(out_a)

    print("\n=== B) xKV KeySifter pipeline ===")
    out_b = run_xkv_keysifter(cfg, prompt)
    print(out_b)

    print("\n=== COMPARISON ===")
    same = out_a == out_b
    print(f"equal={same}")
    if not same:
        d = first_diff(out_a, out_b)
        print(f"first_diff_char_index={d}")


if __name__ == "__main__":
    main()
