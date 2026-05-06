#!/usr/bin/env python3
"""
Ablation study: sweep predict_interval (n) in TokenButler (i=n+neighbor).

Measures decode time and accuracy for n = [1, 2, 4, 8, 16, 32],
all with enable_neighbor_fetch=True at 64K context, sparse_budget=8192
(matching the Table 6 settings).

Produces:
  - results/ablation_predict_interval.csv
  - results/ablation_predict_interval.png

Usage:
    python test/ablation_predict_interval.py                    # full run
    python test/ablation_predict_interval.py --quick            # 15 samples, fewer datasets
    python test/ablation_predict_interval.py --skip-accuracy    # efficiency only
    python test/ablation_predict_interval.py --skip-efficiency  # accuracy only
    python test/ablation_predict_interval.py --load-cache results/ablation_predict_interval.csv
"""

import os
import sys
import csv
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = ROOT / "test" / "eval_acc.py"
RESULTS_DIR = ROOT / "results"

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
PREDICT_INTERVALS = [1, 2, 4, 8, 16, 32]

# All 11 RULER datasets at 64K
ALL_DATASETS = [
    "ruler/niah_single_1",
    "ruler/niah_single_2",
    "ruler/niah_single_3",
    "ruler/niah_multikey_1",
    "ruler/niah_multikey_2",
    "ruler/niah_multiquery",
    "ruler/niah_multivalue",
    "ruler/fwe",
    "ruler/vt",
    "ruler/qa_1",
    "ruler/qa_2",
]

# Quick-mode uses a small subset
QUICK_DATASETS = [
    "ruler/niah_single_1",
    "ruler/niah_multikey_1",
    "ruler/fwe",
    "ruler/qa_1",
]

# Fixed model / eval params (match benchmark_accuracy.py)
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
METHOD = "TokenButler"
DATALEN = 65536
SPARSE_BUDGET = 8192
CHUNK_SIZE = 8
RANK = 160
RANK_K = 96
RANK_V = 144
GROUP_SIZE = 1
DDASH = 16
PRODUCER_FREQ = 4
INTERMEDIATE_DIM = 512
PREDICTOR_PATH = str(ROOT / "L3_8Bi_d16_i512_pf4.pt")

# Efficiency benchmark params
PROMPT_LENGTH = 65536
GEN_LENGTH = 1024


# ---------------------------------------------------------------------------
# Efficiency sweep
# ---------------------------------------------------------------------------
def run_efficiency_sweep(csv_path):
    """Run decode-time benchmarks for each predict_interval + baselines."""
    # Import benchmark_model — add test/ to path same as other scripts
    test_dir = str(ROOT / "test")
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from benchmark_tokenbutler import benchmark_model

    weights_path = PREDICTOR_PATH

    fieldnames = [
        "config", "predict_interval", "enable_neighbor_fetch",
        "decode_time_avg_ms", "decode_tokens_per_sec",
        "prefill_time_s", "memory_gb",
    ]

    # Write header
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    def _append_row(row):
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    def _run(label, **kwargs):
        print(f"\n{'='*60}")
        print(f"  Efficiency: {label}")
        print(f"{'='*60}")
        try:
            result = benchmark_model(**kwargs)
            row = {
                "config": label,
                "predict_interval": kwargs.get("predict_interval", ""),
                "enable_neighbor_fetch": kwargs.get("enable_neighbor_fetch", False),
                "decode_time_avg_ms": f"{result['decode_time_avg'] * 1000:.3f}",
                "decode_tokens_per_sec": f"{result['decode_tokens_per_sec']:.2f}",
                "prefill_time_s": f"{result['prefill_time']:.3f}",
                "memory_gb": f"{result['memory_allocated_gb']:.2f}",
            }
            _append_row(row)
            return result
        except Exception as e:
            print(f"  FAILED: {e}")
            return None

    # --- Reference baselines ---
    _run("Dense",
         attn_mode="full", prompt_length=PROMPT_LENGTH, gen_length=GEN_LENGTH)

    time.sleep(2)

    _run("TokenButler_i1_no_neighbor",
         attn_mode="tokenbutler", prompt_length=PROMPT_LENGTH, gen_length=GEN_LENGTH,
         sparse_budget=SPARSE_BUDGET, predictor_path=weights_path,
         predict_interval=1, enable_neighbor_fetch=False)

    time.sleep(2)

    _run("Oracle_random",
         attn_mode="oracle", prompt_length=PROMPT_LENGTH, gen_length=GEN_LENGTH,
         sparse_budget=SPARSE_BUDGET, oracle_random_indices=True)

    time.sleep(2)

    _run("Oracle_contiguous",
         attn_mode="oracle", prompt_length=PROMPT_LENGTH, gen_length=GEN_LENGTH,
         sparse_budget=SPARSE_BUDGET, oracle_random_indices=False)

    time.sleep(2)

    # --- Interval sweep (all with neighbor fetch) ---
    for n in PREDICT_INTERVALS:
        _run(f"n={n}",
             attn_mode="tokenbutler", prompt_length=PROMPT_LENGTH, gen_length=GEN_LENGTH,
             sparse_budget=SPARSE_BUDGET, predictor_path=weights_path,
             predict_interval=n, enable_neighbor_fetch=True)
        time.sleep(2)


# ---------------------------------------------------------------------------
# Accuracy sweep
# ---------------------------------------------------------------------------
def result_filename(dataset_name):
    ds_short = dataset_name.replace("ruler/", "")
    return (
        f"{ds_short}_{DATALEN}_{METHOD}_b{SPARSE_BUDGET}_c{CHUNK_SIZE}"
        f"_x{GROUP_SIZE}_r{RANK}_k{RANK_K}_v{RANK_V}.jsonl"
    )


def run_accuracy_for_interval(n, num_samples, datasets, output_dir):
    """Run eval_acc.py for one predict_interval value."""
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
        "--predict_interval", str(n),
        "--enable_neighbor_fetch",
    ]
    if num_samples > 0:
        cmd += ["--num_samples", str(num_samples)]

    config_name = f"n={n}"
    print(f"\n{'='*70}")
    print(f"  Accuracy: {config_name}  (predict_interval={n}, neighbor=True)")
    print(f"  Samples per dataset: {'all' if num_samples <= 0 else num_samples}")
    print(f"{'='*70}\n")

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  ERROR: {config_name} exited with code {proc.returncode}")
        return None, elapsed

    # Move result files to our output dir
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

    # Read scores
    scores = {}
    for ds in datasets:
        fname = result_filename(ds)
        fpath = dst_dir / fname
        ds_short = ds.replace("ruler/", "")
        if not fpath.exists():
            scores[ds_short] = None
            continue
        lines = [json.loads(l) for l in open(fpath)]
        if not lines:
            scores[ds_short] = None
            continue
        scores[ds_short] = lines[-1].get("avg_score", 0)

    print(f"  {config_name} done in {timedelta(seconds=int(elapsed))}")
    return scores, elapsed


def run_accuracy_sweep(csv_path, num_samples, datasets, output_dir):
    """Run accuracy benchmarks for each predict_interval."""
    fieldnames = ["config", "predict_interval"] + [
        ds.replace("ruler/", "") for ds in datasets
    ] + ["avg_accuracy"]

    # Write header
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    total_elapsed = 0
    for n in PREDICT_INTERVALS:
        scores, elapsed = run_accuracy_for_interval(
            n, num_samples, datasets, output_dir
        )
        total_elapsed += elapsed

        if scores is None:
            print(f"  Skipping n={n} due to error")
            continue

        valid = [v for v in scores.values() if v is not None]
        avg = sum(valid) / len(valid) if valid else 0

        row = {"config": f"n={n}", "predict_interval": n}
        for ds_short, score in scores.items():
            row[ds_short] = f"{score:.4f}" if score is not None else ""
        row["avg_accuracy"] = f"{avg:.4f}"

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        # Print running summary
        print(f"  n={n}: avg_accuracy={avg:.4f}  ({len(valid)}/{len(datasets)} datasets)")

    print(f"\n  Total accuracy sweep time: {timedelta(seconds=int(total_elapsed))}")


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
def load_cached_results(csv_path):
    """Load results from a combined CSV (efficiency + accuracy rows)."""
    efficiency = {}  # config -> decode_time_avg_ms
    accuracy = {}    # config -> avg_accuracy

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            config = row.get("config", "")
            if "decode_time_avg_ms" in row and row["decode_time_avg_ms"]:
                efficiency[config] = float(row["decode_time_avg_ms"])
            if "avg_accuracy" in row and row["avg_accuracy"]:
                accuracy[config] = float(row["avg_accuracy"])

    return efficiency, accuracy


def merge_csvs(eff_csv, acc_csv, combined_csv):
    """Merge efficiency and accuracy CSVs into one combined file."""
    rows = []

    if eff_csv.exists():
        with open(eff_csv, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)

    if acc_csv.exists():
        # Merge accuracy into matching efficiency rows by config name
        acc_by_config = {}
        with open(acc_csv, newline="") as f:
            for row in csv.DictReader(f):
                acc_by_config[row["config"]] = row

        for r in rows:
            acc_row = acc_by_config.pop(r["config"], None)
            if acc_row:
                r.update(acc_row)

        # Add any accuracy-only rows
        for row in acc_by_config.values():
            rows.append(row)

    if not rows:
        return

    all_keys = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    with open(combined_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(efficiency, accuracy, output_path):
    """
    Dual y-axis plot.
    - X: predict_interval (log scale)
    - Left Y (blue): decode time per token (ms)
    - Right Y (red): average accuracy
    - Horizontal dashed lines for reference baselines
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract sweep data points (n=1, n=2, ...)
    intervals = []
    eff_values = []
    acc_values = []

    for n in PREDICT_INTERVALS:
        key = f"n={n}"
        if key in efficiency:
            intervals.append(n)
            eff_values.append(efficiency[key])
            acc_values.append(accuracy.get(key))

    if not intervals:
        print("No sweep data to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Left y-axis: decode time (blue)
    color_eff = "#1f77b4"
    ax1.set_xlabel("predict_interval (n)", fontsize=12)
    ax1.set_ylabel("Avg decode time per token (ms)", fontsize=12, color=color_eff)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(PREDICT_INTERVALS)
    ax1.set_xticklabels([str(n) for n in PREDICT_INTERVALS])

    line1, = ax1.plot(intervals, eff_values, "o-", color=color_eff, linewidth=2,
                      markersize=8, label="Decode time (ms)", zorder=5)
    ax1.tick_params(axis="y", labelcolor=color_eff)

    # Reference baselines (horizontal dashed lines)
    baseline_styles = {
        "Dense": {"color": "gray", "linestyle": "--", "linewidth": 1.5},
        "TokenButler_i1_no_neighbor": {"color": color_eff, "linestyle": ":", "linewidth": 1.5},
        "Oracle_random": {"color": "green", "linestyle": "--", "linewidth": 1.2},
        "Oracle_contiguous": {"color": "green", "linestyle": ":", "linewidth": 1.2},
    }

    baseline_labels_added = []
    for bname, style in baseline_styles.items():
        if bname in efficiency:
            label = bname.replace("_", " ")
            ax1.axhline(y=efficiency[bname], label=f"{label} ({efficiency[bname]:.1f}ms)",
                        alpha=0.7, **style)
            baseline_labels_added.append(bname)

    # Right y-axis: accuracy (red)
    has_accuracy = any(v is not None for v in acc_values)
    if has_accuracy:
        color_acc = "#d62728"
        ax2 = ax1.twinx()
        ax2.set_ylabel("Average accuracy (11 RULER datasets)", fontsize=12, color=color_acc)

        valid_intervals = [intervals[i] for i, v in enumerate(acc_values) if v is not None]
        valid_acc = [v for v in acc_values if v is not None]

        line2, = ax2.plot(valid_intervals, valid_acc, "s-", color=color_acc, linewidth=2,
                          markersize=8, label="Accuracy", zorder=5)
        ax2.tick_params(axis="y", labelcolor=color_acc)

        # Combine legends from both axes
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        # Add baseline legend entries from ax1
        ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
        for h, l in zip(ax1_handles, ax1_labels):
            if h is not line1:
                lines.append(h)
                labels.append(l)
        ax2.legend(lines, labels, loc="upper left", fontsize=9)
    else:
        ax1.legend(loc="upper left", fontsize=9)

    ax1.grid(axis="both", alpha=0.3, linestyle="--")
    ax1.set_title("Predict Interval Ablation (enable_neighbor_fetch=True, 64K context)",
                  fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ablation study: predict_interval sweep for TokenButler"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 15 samples, fewer datasets")
    parser.add_argument("--skip-accuracy", action="store_true",
                        help="Skip accuracy benchmarks (efficiency only)")
    parser.add_argument("--skip-efficiency", action="store_true",
                        help="Skip efficiency benchmarks (accuracy only)")
    parser.add_argument("--load-cache", type=str, default=None,
                        help="Load results from a cached CSV and plot only")
    parser.add_argument("--num-samples", type=int, default=-1,
                        help="Override number of samples per dataset (-1 = all, default 96)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    combined_csv = RESULTS_DIR / "ablation_predict_interval.csv"
    plot_path = RESULTS_DIR / "ablation_predict_interval.png"

    # --- Load-cache mode: just plot ---
    if args.load_cache:
        print(f"Loading cached results from {args.load_cache}")
        efficiency, accuracy = load_cached_results(args.load_cache)
        print(f"  Efficiency entries: {len(efficiency)}")
        print(f"  Accuracy entries: {len(accuracy)}")
        plot_results(efficiency, accuracy, plot_path)
        return

    # --- Determine sample count ---
    if args.quick:
        num_samples = 15
        datasets = QUICK_DATASETS
    else:
        num_samples = args.num_samples if args.num_samples > 0 else 96
        datasets = ALL_DATASETS

    ts = time.strftime("%Y%m%d_%H%M%S")
    eff_csv = RESULTS_DIR / f"ablation_pi_efficiency_{ts}.csv"
    acc_csv = RESULTS_DIR / f"ablation_pi_accuracy_{ts}.csv"

    print(f"{'='*70}")
    print(f"  Predict Interval Ablation Study")
    print(f"{'='*70}")
    print(f"  Intervals:  {PREDICT_INTERVALS}")
    print(f"  Context:    {DATALEN // 1024}K tokens")
    print(f"  Budget:     {SPARSE_BUDGET}")
    print(f"  Neighbor:   True (all sweep configs)")
    print(f"  Datasets:   {len(datasets)}")
    print(f"  Samples:    {'all' if num_samples <= 0 else num_samples} per dataset")
    print(f"  Quick:      {args.quick}")
    print(f"  Skip eff:   {args.skip_efficiency}")
    print(f"  Skip acc:   {args.skip_accuracy}")
    print()

    # --- Efficiency sweep ---
    if not args.skip_efficiency:
        print(f"\n{'#'*70}")
        print(f"  PHASE 1: Efficiency Sweep")
        print(f"{'#'*70}\n")
        run_efficiency_sweep(eff_csv)
    else:
        print("Skipping efficiency sweep.")

    # --- Accuracy sweep ---
    acc_output_dir = RESULTS_DIR / f"ablation_pi_acc_{ts}"
    if not args.skip_accuracy:
        print(f"\n{'#'*70}")
        print(f"  PHASE 2: Accuracy Sweep")
        print(f"{'#'*70}\n")
        acc_output_dir.mkdir(parents=True, exist_ok=True)
        run_accuracy_sweep(acc_csv, num_samples, datasets, acc_output_dir)
    else:
        print("Skipping accuracy sweep.")

    # --- Merge and plot ---
    merge_csvs(eff_csv, acc_csv, combined_csv)
    print(f"\nCombined results: {combined_csv}")

    efficiency, accuracy = load_cached_results(combined_csv)
    plot_results(efficiency, accuracy, plot_path)

    # --- Summary table ---
    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    print(f"  {'n':>4s}  {'Decode (ms)':>12s}  {'Accuracy':>10s}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*10}")
    for n in PREDICT_INTERVALS:
        key = f"n={n}"
        eff_str = f"{efficiency[key]:.2f}" if key in efficiency else "N/A"
        acc_str = f"{accuracy[key]:.4f}" if key in accuracy else "N/A"
        print(f"  {n:>4d}  {eff_str:>12s}  {acc_str:>10s}")

    if "Dense" in efficiency:
        print(f"\n  Dense baseline: {efficiency['Dense']:.2f} ms/tok")
    if "TokenButler_i1_no_neighbor" in efficiency:
        print(f"  KS i=1 (no neighbor): {efficiency['TokenButler_i1_no_neighbor']:.2f} ms/tok")
    print()


if __name__ == "__main__":
    main()
