#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

# Kernels to match (substring match on Name column)
DEFAULT_PATTERNS = [
    "gather_copy_var_midpoint_BP",
    "gather_copy_d2d",
    "unrolled_elementwise_kernel",
    "Kernel2",
    "apply_rotary_pos_emb_kernel_push_cache_opt",
    "flash_fwd_splitkv_kernel",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract selected kernel times from nsys --report cuda_gpu_kern_sum CSVs")
    p.add_argument("inputs", nargs='*', help="CSV files or directories to search. If empty, uses --path")
    p.add_argument("--path", default=".", help="Root path to search for CSVs when no inputs provided (default: .)")
    p.add_argument("--recursive", action='store_true', help="Recursively search directories for *_cuda_gpu_kern_sum.csv")
    p.add_argument("--patterns", nargs='*', default=DEFAULT_PATTERNS,
                   help="Kernel name substrings to match (default: common kernels)")
    p.add_argument("--out", default="kernel_times.csv", help="Output CSV (transposed) path")
    return p.parse_args()

# python3 scripts/extract_nsys_kernel_times.py --path . --out kernel_times.csv

def discover_csvs(inputs: List[str], root_path: str, recursive: bool) -> List[str]:
    files: List[str] = []
    candidates: List[str] = inputs if inputs else [root_path]
    for p in candidates:
        if os.path.isfile(p) and p.lower().endswith('cuda_gpu_kern_sum.csv'):
            files.append(p)
        elif os.path.isdir(p):
            if recursive:
                for dirpath, _, filenames in os.walk(p):
                    for fn in filenames:
                        if fn.endswith('_cuda_gpu_kern_sum.csv'):
                            files.append(os.path.join(dirpath, fn))
            else:
                for fn in os.listdir(p):
                    fp = os.path.join(p, fn)
                    if os.path.isfile(fp) and (fn.endswith('_cuda_gpu_kern_sum.csv')):
                        files.append(fp)
        else:
            # Ignore non-existent paths
            continue
    # Dedup and keep stable order
    seen = set()
    result: List[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result


def extract_from_file(path: str, patterns: List[str]) -> Dict[str, float]:
    """Return mapping pattern -> mean of 'Avg (ns)' across matched rows."""
    sum_avg_ns: Dict[str, float] = {pat: 0.0 for pat in patterns}
    counts: Dict[str, int] = {pat: 0 for pat in patterns}

    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('Name', '')
            if not name:
                continue
            try:
                avg_ns = float(row.get('Avg (ns)', '0'))
            except ValueError:
                # Skip malformed rows
                continue

            for pat in patterns:
                if pat in name:
                    sum_avg_ns[pat] += avg_ns
                    counts[pat] += 1
                    break
    # Compute simple mean of 'Avg (ns)' per pattern
    result: Dict[str, float] = {}
    for pat in patterns:
        cnt = counts[pat]
        result[pat] = (sum_avg_ns[pat] / cnt) if cnt > 0 else 0.0

    return result


def main():
    args = parse_args()
    patterns = args.patterns

    csv_files = discover_csvs(args.inputs, args.path, args.recursive)
    if not csv_files:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(1)

    # For pivot only
    pivot_vals: Dict[str, Dict[str, float]] = {pat: {} for pat in patterns}

    for csv_path in csv_files:
        avg_map = extract_from_file(csv_path, patterns)
        base = os.path.basename(csv_path)
        for pat in patterns:
            avg_ns = avg_map.get(pat, 0.0)
            avg_ms = avg_ns / 1e3
            pivot_vals[pat][base] = avg_ms

    # Write pivot CSV: rows=patterns, cols=files (avg_time_ms)
    files_sorted = sorted({os.path.basename(p) for p in csv_files})
    pivot_header = ["pattern (avg_ms)"] + files_sorted
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(pivot_header)
        for pat in patterns:
            row = [pat]
            for fn in files_sorted:
                val = pivot_vals.get(pat, {}).get(fn, 0.0)
                row.append(f"{val:.3f}")
            w.writerow(row)


if __name__ == "__main__":
    main()
