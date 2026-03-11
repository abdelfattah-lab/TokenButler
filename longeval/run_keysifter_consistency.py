#!/usr/bin/env python3
"""Run a small longeval-style sample through the custom xKV `models.Llama` KeySifter
pipeline and check for deterministic / consistent outputs across runs.

Why this exists:
- `xKV/longeval` is built around HuggingFace `model.generate()` plus KV-compression patches.
- KeySifter in this repo is implemented in the custom `xKV/models/*` pipeline and uses a
  separate predictor checkpoint (e.g., `Llama_31_8bi_GQA_dDash16.pt`).

This script reuses longeval's dataset/prompt formatting, but uses the custom model path.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch

# Ensure repo imports work when running as a script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from xKV.longeval.data.dataset import Dataset  # noqa: E402
from xKV.models import Llama  # noqa: E402


@dataclass
class RunConfig:
    model_path: str
    predictor_path: str
    producer_frequency: int
    dDash: int
    max_seq_len: int
    dataset: str
    datalen: int
    num_samples: int
    sample_idx: int
    gen_len: int | None
    temperature: float
    runs: int
    seed: int
    device: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_input_ids(tokenized_prompt) -> torch.Tensor:
    # longeval Dataset stores prompts as tokenized objects; support common shapes.
    if isinstance(tokenized_prompt, dict):
        input_ids = tokenized_prompt.get("input_ids")
    else:
        input_ids = getattr(tokenized_prompt, "input_ids", None)
    if input_ids is None:
        raise TypeError(f"Unsupported tokenized prompt type: {type(tokenized_prompt)}")

    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Base Llama weights path (HF or local, per xKV/models/Llama)")
    parser.add_argument("--predictor-path", required=True, help="KeySifter predictor checkpoint (e.g. xKV/Llama_31_8bi_GQA_dDash16.pt)")
    parser.add_argument("--producer-frequency", type=int, default=16)
    parser.add_argument("--dDash", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=32768)

    parser.add_argument("--dataset", default="long_bench/narrativeqa")
    parser.add_argument("--datalen", type=int, default=32768)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--gen-len", type=int, default=None, help="Override dataset generation length")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    cfg = RunConfig(
        model_path=args.model_path,
        predictor_path=args.predictor_path,
        producer_frequency=args.producer_frequency,
        dDash=args.dDash,
        max_seq_len=args.max_seq_len,
        dataset=args.dataset,
        datalen=args.datalen,
        num_samples=args.num_samples,
        sample_idx=args.sample_idx,
        gen_len=args.gen_len,
        temperature=args.temperature,
        runs=args.runs,
        seed=args.seed,
        device=args.device,
    )

    seed_everything(cfg.seed)

    llm = Llama(
        model_name=cfg.model_path,
        max_length=cfg.max_seq_len,
        device=cfg.device,
        attn_mode="keysifter",
        predictor_path=cfg.predictor_path,
        producer_frequency=cfg.producer_frequency,
        dDash=cfg.dDash,
    )

    # longeval Dataset needs a HF-like tokenizer; xKV Llama exposes `tokenizer` compatible enough.
    dataset = Dataset(cfg.dataset, llm.tokenizer, datalen=cfg.datalen, num_samples=cfg.num_samples)

    if not hasattr(dataset, "tokenized_prompts") or len(dataset.tokenized_prompts) == 0:
        raise RuntimeError("Dataset produced no tokenized prompts")

    idx = cfg.sample_idx
    if idx < 0 or idx >= len(dataset.tokenized_prompts):
        raise ValueError(f"sample-idx {idx} out of range (0..{len(dataset.tokenized_prompts) - 1})")

    tokenized_prompt = dataset.tokenized_prompts[idx]
    input_ids = _to_input_ids(tokenized_prompt).to(cfg.device)

    gen_len = cfg.gen_len if cfg.gen_len is not None else getattr(dataset, "gen_len", 256)

    outputs: list[str] = []
    for run_i in range(cfg.runs):
        # Reset caches to ensure each run starts from identical state.
        if hasattr(llm, "kv_cache") and hasattr(llm.kv_cache, "clear"):
            llm.kv_cache.clear()

        seed_everything(cfg.seed)

        with torch.inference_mode():
            text = llm.generate(
                input_ids,
                gen_len=gen_len,
                temperature=cfg.temperature,
                verbose=False,
            )[0]
        outputs.append(text)

        print(f"\n=== RUN {run_i + 1}/{cfg.runs} ===")
        print(text)

    all_equal = all(o == outputs[0] for o in outputs[1:])
    print("\n=== CONSISTENCY ===")
    print(f"dataset={cfg.dataset} datalen={cfg.datalen} gen_len={gen_len} producer_frequency={cfg.producer_frequency}")
    print(f"all_outputs_equal={all_equal}")
    if not all_equal:
        for i in range(1, len(outputs)):
            if outputs[i] != outputs[0]:
                # Print a small prefix diff hint without pulling in heavy deps.
                a, b = outputs[0], outputs[i]
                first_diff = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), None)
                print(f"first_diff_run0_vs_run{i}={first_diff}")
                break


if __name__ == "__main__":
    main()
