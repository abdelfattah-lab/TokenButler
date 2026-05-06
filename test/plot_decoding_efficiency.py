#!/usr/bin/env python3
"""Plot decoding latency vs context length from the efficiency-sweep CSV.

Reads the CSV produced by `test/run_missing_configs.py`
(`test/output/efficiency_budget8K_1M/decoding_time_vs_context.csv`) and
produces two PDFs:
  - decoding_performance.pdf       — GPU-only contexts (<= 128K)
  - decoding_performance_cpu.pdf   — CPU-offload contexts (>= 256K)
Each compares Dense, TokenButler (multiple intervals), and Oracle baselines.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

CPU_OFFLOAD_THRESHOLD = 131072

STYLE = {
    "Dense":                ("#1f77b4", "-",  "o"),
    "TokenButler (i=1)":    ("#ff7f0e", "-",  "s"),
    "TokenButler (i=2+nb)": ("#ff9d3a", "-",  "D"),
    "TokenButler (i=4+nb)": ("#ffb366", "-",  "^"),
    "TokenButler (i=8+nb)": ("#ffc78f", "-",  "v"),
    "TokenButler (i=16+nb)":("#ffd9b3", "-",  "p"),
    "Oracle (random)":      ("#d62728", ":",  "x"),
    "Oracle (contiguous)":  ("#9467bd", ":",  "+"),
    "Oracle (random, i=16)":("#8c564b", "--", "*"),
}


def load(csv_path: Path):
    series = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "success":
                continue
            try:
                ctx = int(row["context_length"])
                ms = float(row.get("decode_time_ms") or row.get("decode_ms"))
            except (TypeError, ValueError):
                continue
            label = row["label"]
            if label.startswith("KeySifter"):
                continue  # legacy rows from pre-rename runs; ignore for plotting
            series[label].append((ctx, ms))
    return {k: sorted(v) for k, v in series.items()}


def render(series, contexts, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, points in series.items():
        xs = [c for c, _ in points if c in contexts]
        ys = [m for c, m in points if c in contexts]
        if not xs:
            continue
        color, ls, marker = STYLE.get(label, ("#444", "-", "."))
        ax.plot(xs, ys, color=color, linestyle=ls, marker=marker, markersize=7, linewidth=2, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Decode latency (ms/token)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    sorted_ctx = sorted(contexts)
    ax.set_xticks(sorted_ctx)
    ax.set_xticklabels([f"{c//1024}K" if c < 1_000_000 else "1M" for c in sorted_ctx])
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"saved {out_path} (+ .png)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="test/output/efficiency_budget8K_1M/decoding_time_vs_context.csv")
    p.add_argument("--output-dir", default="paper_plots")
    args = p.parse_args()

    series = load(Path(args.csv))
    if not series:
        raise SystemExit(f"No successful rows in {args.csv}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ctx = sorted({c for pts in series.values() for c, _ in pts})
    gpu_ctx = {c for c in all_ctx if c <= CPU_OFFLOAD_THRESHOLD}
    cpu_ctx = {c for c in all_ctx if c >= CPU_OFFLOAD_THRESHOLD}

    if gpu_ctx:
        render(series, gpu_ctx, out_dir / "decoding_performance.pdf",
               "Decoding latency vs context (GPU-resident)")
    if cpu_ctx:
        render(series, cpu_ctx, out_dir / "decoding_performance_cpu.pdf",
               "Decoding latency vs context (CPU-offload)")


if __name__ == "__main__":
    main()
