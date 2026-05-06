#!/usr/bin/env python3
"""
Accuracy benchmark for TokenButler configurations.

Runs three configs (baseline, interval=8, interval=8+neighbor) across RULER
datasets and prints a comparison table.

Usage:
    python test/benchmark_accuracy.py                       # full (all samples, all datasets)
    python test/benchmark_accuracy.py --quick                # quick (15 samples/dataset)
    python test/benchmark_accuracy.py --num_samples 30       # custom sample count
    python test/benchmark_accuracy.py --datasets niah         # NIAH tasks only
    python test/benchmark_accuracy.py --datasets aggregation  # aggregation tasks only
    python test/benchmark_accuracy.py --datasets all          # all RULER tasks (default)
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = ROOT / "test" / "eval_acc.py"

# Dataset groups -- all available RULER tasks at 32K
DATASET_GROUPS = {
    "niah": [
        "ruler/niah_single_1",
        "ruler/niah_single_2",
        "ruler/niah_single_3",
        "ruler/niah_multikey_1",
        "ruler/niah_multikey_2",
        "ruler/niah_multiquery",
        "ruler/niah_multivalue",
    ],
    # niah_multiturn excluded: uses multi-turn schema (queries/answers)
    # that the dataset loader doesn't support for ruler tasks
    # "niah_multiturn": [
    #     "ruler/niah_multiturn_1",
    #     "ruler/niah_multiturn_2",
    # ],
    "aggregation": [
        "ruler/fwe",
        "ruler/vt",
    ],
    "qa": [
        "ruler/qa_1",
        "ruler/qa_2",
    ],
}

# Flattened default: all datasets
ALL_DATASETS = [ds for group in DATASET_GROUPS.values() for ds in group]

CONFIGS = {
    "baseline": {
        "predict_interval": 1,
        "enable_neighbor_fetch": False,
    },
    "interval8": {
        "predict_interval": 8,
        "enable_neighbor_fetch": False,
    },
    "interval8_neighbor": {
        "predict_interval": 8,
        "enable_neighbor_fetch": True,
    },
}

# Fixed model/eval params
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
METHOD = "TokenButler"
DATALEN = 32768
SPARSE_BUDGET = 2048
CHUNK_SIZE = 8
RANK = 160
RANK_K = 96
RANK_V = 144
GROUP_SIZE = 1
DDASH = 16
PRODUCER_FREQ = 4
INTERMEDIATE_DIM = 512
PREDICTOR_PATH = str(ROOT / "L3_8Bi_d16_i512_pf4.pt")


def result_filename(dataset_name):
    ds_short = dataset_name.replace("ruler/", "")
    return (
        f"{ds_short}_{DATALEN}_{METHOD}_b{SPARSE_BUDGET}_c{CHUNK_SIZE}"
        f"_x{GROUP_SIZE}_r{RANK}_k{RANK_K}_v{RANK_V}.jsonl"
    )


def run_config(config_name, config, num_samples, output_dir, datasets):
    """Run eval_acc.py for one config across all datasets."""
    dataset_str = ",".join(datasets)

    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--model_name", MODEL_NAME,
        "--method", METHOD,
        "--datalen", str(DATALEN),
        "--dataset_name", dataset_str,
        "--sparse_budget", str(SPARSE_BUDGET),
        "--chunk_size", str(CHUNK_SIZE),
        "--rank", str(RANK),
        "--rank_k", str(RANK_K),
        "--rank_v", str(RANK_V),
        "--group_size", str(GROUP_SIZE),
        "--dDash", str(DDASH),
        "--producer_frequency", str(PRODUCER_FREQ),
        "--tokenbutler_intermediate_dim", str(INTERMEDIATE_DIM),
        "--predictor_path", PREDICTOR_PATH,
        "--predict_interval", str(config["predict_interval"]),
    ]
    if num_samples > 0:
        cmd += ["--num_samples", str(num_samples)]
    if config["enable_neighbor_fetch"]:
        cmd.append("--enable_neighbor_fetch")

    print(f"\n{'='*70}")
    print(f"  Config: {config_name}  (predict_interval={config['predict_interval']}, "
          f"neighbor={config['enable_neighbor_fetch']})")
    print(f"  Samples per dataset: {'all' if num_samples <= 0 else num_samples}")
    print(f"{'='*70}\n")

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"\n  ERROR: {config_name} exited with code {proc.returncode}")
        return False, elapsed

    # Move results from the default output path to our output dir
    model_short = MODEL_NAME.split("/")[-1]
    src_dir = ROOT / "archive" / model_short / "ruler"
    dst_dir = output_dir / config_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        fname = result_filename(ds)
        src = src_dir / fname
        dst = dst_dir / fname
        if src.exists():
            src.rename(dst)
        else:
            print(f"  WARNING: expected result file not found: {src}")

    print(f"\n  {config_name} done in {timedelta(seconds=int(elapsed))}")
    return True, elapsed


def read_scores(output_dir, datasets):
    """Read scores from all config result directories."""
    results = {}
    for config_name in CONFIGS:
        results[config_name] = {}
        cfg_dir = output_dir / config_name
        for ds in datasets:
            fname = result_filename(ds)
            fpath = cfg_dir / fname
            ds_short = ds.replace("ruler/", "")
            if not fpath.exists():
                results[config_name][ds_short] = (0, None)
                continue
            lines = [json.loads(l) for l in open(fpath)]
            if not lines:
                results[config_name][ds_short] = (0, None)
                continue
            # The evaluator writes cumulative correct lists, so the last
            # line's avg_score is the final score over all samples.
            n = len(lines)
            avg = lines[-1].get("avg_score", 0)
            results[config_name][ds_short] = (n, avg)
    return results


def print_table(results, total_elapsed, datasets):
    """Print a formatted comparison table."""
    ds_names = [ds.replace("ruler/", "") for ds in datasets]
    config_names = list(CONFIGS.keys())
    col_labels = ["baseline (i=1)", "interval=8", "i=8 + neighbor"]

    # Header
    print(f"\n{'='*80}")
    print(f"  TokenButler Accuracy Benchmark Results")
    print(f"{'='*80}\n")

    header = f"| {'Dataset':<18} |"
    for label in col_labels:
        header += f" {label:>18} |"
    print(header)
    print("|" + "-" * 20 + "|" + (("-" * 20 + "|") * len(col_labels)))

    # Rows
    cfg_avgs = {cfg: [] for cfg in config_names}
    for ds in ds_names:
        row = f"| {ds:<18} |"
        for cfg in config_names:
            n, score = results[cfg].get(ds, (0, None))
            if score is not None:
                row += f" {score:>11.4f}  ({n:>2}) |"
                cfg_avgs[cfg].append(score)
            else:
                row += f" {'N/A':>18} |"
        print(row)

    # Average row
    print("|" + "-" * 20 + "|" + (("-" * 20 + "|") * len(col_labels)))
    row = f"| {'AVERAGE':<18} |"
    for cfg in config_names:
        avgs = cfg_avgs[cfg]
        if avgs:
            row += f" {sum(avgs)/len(avgs):>11.4f}       |"
        else:
            row += f" {'N/A':>18} |"
    print(row)

    print(f"\n  Total time: {timedelta(seconds=int(total_elapsed))}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark TokenButler accuracy across configurations")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 15 samples per dataset")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Number of samples per dataset (-1 = all)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated configs to run (default: all). "
                             "Options: baseline,interval8,interval8_neighbor")
    parser.add_argument("--datasets", type=str, default="all",
                        help="Dataset group to run (default: all). "
                             "Options: all, niah, niah_multiturn, aggregation, qa, "
                             "or comma-separated list of ruler/ dataset names")
    args = parser.parse_args()

    num_samples = args.num_samples
    if args.quick:
        num_samples = 15

    # Which datasets
    ds_arg = args.datasets.strip()
    if ds_arg == "all":
        datasets = ALL_DATASETS
    elif ds_arg in DATASET_GROUPS:
        datasets = DATASET_GROUPS[ds_arg]
    else:
        # Treat as comma-separated list; add ruler/ prefix if missing
        datasets = []
        for d in ds_arg.split(","):
            d = d.strip()
            if not d.startswith("ruler/"):
                d = "ruler/" + d
            datasets.append(d)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_short = MODEL_NAME.split("/")[-1]
        tag = "quick" if args.quick else (f"n{num_samples}" if num_samples > 0 else "full")
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "archive" / model_short / f"benchmark_{tag}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Which configs
    if args.configs:
        selected = [c.strip() for c in args.configs.split(",")]
        configs_to_run = {k: v for k, v in CONFIGS.items() if k in selected}
    else:
        configs_to_run = CONFIGS

    print(f"Output:   {output_dir}")
    print(f"Configs:  {list(configs_to_run.keys())}")
    print(f"Datasets: {len(datasets)} — {[d.replace('ruler/', '') for d in datasets]}")
    print(f"Samples:  {'all' if num_samples <= 0 else num_samples} per dataset")

    total_elapsed = 0
    for config_name, config in configs_to_run.items():
        ok, elapsed = run_config(config_name, config, num_samples, output_dir, datasets)
        total_elapsed += elapsed
        if not ok:
            print(f"Aborting due to error in {config_name}")
            sys.exit(1)

    # Read and display
    results = read_scores(output_dir, datasets)
    print_table(results, total_elapsed, datasets)
    # Save the table data as MD file
    md_path = output_dir / "results.md"
    with open(md_path, "w") as f:
        f.write(f"# TokenButler Accuracy Benchmark Results\n\n")
        f.write(f"**Model:** {MODEL_NAME}\n\n")
        f.write(f"**Method:** {METHOD}\n\n")
        f.write(f"**Datasets:** {', '.join([ds.replace('ruler/', '') for ds in datasets])}\n\n")
        f.write(f"**Samples per dataset:** {'all' if num_samples <= 0 else num_samples}\n\n")
        f.write(f"**Total time:** {timedelta(seconds=int(total_elapsed))}\n\n")

        # Table header
        col_labels = ["baseline (i=1)", "interval=8", "i=8 + neighbor"]
        header = f"| {'Dataset':<18} |"
        for label in col_labels:
            header += f" {label:>18} |"
        f.write(header + "\n")
        f.write("|" + "-" * 20 + "|" + (("-" * 20 + "|") * len(col_labels)) + "\n")

        # Table rows
        cfg_avgs = {cfg: [] for cfg in CONFIGS}
        for ds in [ds.replace("ruler/", "") for ds in datasets]:
            row = f"| {ds:<18} |"
            for cfg in CONFIGS:
                n, score = results[cfg].get(ds, (0, None))
                if score is not None:
                    row += f" {score:>11.4f}  ({n:>2}) |"
                    cfg_avgs[cfg].append(score)
                else:
                    row += f" {'N/A':>18} |"
            f.write(row + "\n")

        # Average row
        f.write("|" + "-" * 20 + "|" + (("-" * 20 + "|") * len(col_labels)) + "\n")
        row = f"| {'AVERAGE':<18} |"
        for cfg in CONFIGS:
            avgs = cfg_avgs[cfg]
            if avgs:
                row += f" {sum(avgs)/len(avgs):>11.4f}       |"
            else:
                row += f" {'N/A':>18} |"
        f.write(row + "\n")

if __name__ == "__main__":
    main()
