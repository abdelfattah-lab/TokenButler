#!/usr/bin/env python3
"""
Aggregate .*log files into a CSV splitting Throughput and Memory into separate columns per method.

Columns:
- length, bsz, rank
- Memory (GB): Baseline (GB), ShadowKV (GB), xKey-1 (GB), xKey-2 (GB), xKey-4 (GB)
- Throughput (token/s): Baseline (token/s), ShadowKV (token/s), xKey-1 (token/s), xKey-2 (token/s), xKey-4 (token/s)

If Baseline OOM is reported for a combination, the Baseline columns remain empty.
If any method is missing for a combination, the cells are left blank.

Usage:
    python scripts/extract_e2e_throughput.py --logs logs --out e2e.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

RecordKey = Tuple[int, int, Optional[int]]  # (length, bsz, rank)


# Regex patterns for parsing lines
P_SHADOW = re.compile(
    r"\[Shadowkv,\s*prompt_len=(?P<len>\d+),\s*bsz=(?P<bsz>\d+),\s*rank=(?P<rank>\d+)\]\s*Peak Memory:\s*(?P<mem>[0-9.]+)\s*GB,\s*Throughput:\s*(?P<thr>[0-9.]+)\s*token/s",
    re.IGNORECASE,
)

P_BASELINE = re.compile(
    r"\[Baseline,\s*prompt_len=(?P<len>\d+),\s*bsz=(?P<bsz>\d+)\]\s*,?\s*Peak Memory:\s*(?P<mem>[0-9.]+)\s*GB,\s*Throughput:\s*(?P<thr>[0-9.]+)\s*token/s",
    re.IGNORECASE,
)

P_BASELINE_OOM = re.compile(
    r"Baseline OOM for prompt_len=(?P<len>\d+),\s*bsz=(?P<bsz>\d+)", re.IGNORECASE
)

P_XKEY = re.compile(
    r"\[xkey,\s*prompt_len=(?P<len>\d+),\s*bsz=(?P<bsz>\d+),\s*gs=(?P<gs>\d+),\s*rank_k=(?P<rank>\d+)\]\s*Peak Memory:\s*(?P<mem>[0-9.]+)\s*GB,\s*Throughput:\s*(?P<thr>[0-9.]+)\s*token/s",
    re.IGNORECASE,
)

# OOM lines for methods
P_SHADOW_OOM = re.compile(
    r"ShadowKV OOM for prompt_len=(?P<len>\d+),\s*bsz=(?P<bsz>\d+),\s*rank=(?P<rank>\d+)",
    re.IGNORECASE,
)
P_XKEY_OOM = re.compile(
    r"xKey OOM for prompt_len=(?P<len>\d+),\s*bsz=(?P<bsz>\d+),\s*gs=(?P<gs>\d+),\s*rank_k=(?P<rank>\d+)",
    re.IGNORECASE,
)


def parse_logs(log_dir: str) -> Dict[RecordKey, Dict[str, Dict[str, float]]]:
    # results[(length, bsz, rank)][method] = {"thr": float, "mem": float, "oom": bool}
    results: Dict[RecordKey, Dict[str, Dict[str, float]]] = defaultdict(dict)

    # Track Baseline OOMs
    baseline_oom: set[Tuple[int, int]] = set()

    files = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    m = P_BASELINE_OOM.search(line)
                    if m:
                        length = int(m.group("len"))
                        bsz = int(m.group("bsz"))
                        baseline_oom.add((length, bsz))
                        # Store OOM under rank=None row; mark explicit OOM
                        key = (length, bsz, None)
                        results[key]["Baseline"] = {"oom": True}
                        continue

                    m = P_BASELINE.search(line)
                    if m:
                        length = int(m.group("len"))
                        bsz = int(m.group("bsz"))
                        mem = float(m.group("mem"))
                        thr = float(m.group("thr"))
                        key = (length, bsz, None)
                        results[key]["Baseline"] = {"thr": thr, "mem": mem}
                        continue

                    # ShadowKV OOM
                    m = P_SHADOW_OOM.search(line)
                    if m:
                        length = int(m.group("len"))
                        bsz = int(m.group("bsz"))
                        rank = int(m.group("rank"))
                        key = (length, bsz, rank)
                        results[key]["ShadowKV"] = {"oom": True}
                        continue

                    m = P_SHADOW.search(line)
                    if m:
                        length = int(m.group("len"))
                        bsz = int(m.group("bsz"))
                        rank = int(m.group("rank"))
                        mem = float(m.group("mem"))
                        thr = float(m.group("thr"))
                        key = (length, bsz, rank)
                        results[key]["ShadowKV"] = {"thr": thr, "mem": mem}
                        # If there's an OOM recorded for this (length, bsz) and no explicit baseline yet, mark presence
                        if (length, bsz) in baseline_oom and (length, bsz, None) not in results:
                            results[(length, bsz, None)]["Baseline"] = {"oom": True}
                        continue

                    m = P_XKEY.search(line)
                    if m:
                        length = int(m.group("len"))
                        bsz = int(m.group("bsz"))
                        gs = int(m.group("gs"))
                        rank_k = int(m.group("rank"))
                        mem = float(m.group("mem"))
                        thr = float(m.group("thr"))
                        # Use effective rank = rank_k / gs (e.g., xKey-2 → rank/2, xKey-4 → rank/4)
                        eff_rank = max(1, rank_k // max(1, gs))
                        key = (length, bsz, eff_rank)
                        label = f"xKey-{gs}"
                        results[key][label] = {"thr": thr, "mem": mem}
                        continue

                    # xKey OOM
                    m = P_XKEY_OOM.search(line)
                    if m:
                        length = int(m.group("len"))
                        bsz = int(m.group("bsz"))
                        gs = int(m.group("gs"))
                        rank_k = int(m.group("rank"))
                        eff_rank = max(1, rank_k // max(1, gs))
                        key = (length, bsz, eff_rank)
                        label = f"xKey-{gs}"
                        results[key][label] = {"oom": True}
                        continue
        except Exception as e:
            print(f"Error reading {path}: {e}")

    return results


def to_rows(results: Dict[RecordKey, Dict[str, Dict[str, float]]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    # Merge Baseline (rank=None) into rank=64 row per (length, bsz)
    merged: Dict[RecordKey, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for (length, bsz, rank), methods in results.items():
        if rank is None:
            target_key = (length, bsz, 64)
            # Merge only Baseline into the 64 row
            if "Baseline" in methods:
                merged[target_key]["Baseline"] = methods["Baseline"]
        else:
            # Copy over all methods for this exact rank
            dst = merged[(length, bsz, rank)]
            for mname, mvals in methods.items():
                dst[mname] = mvals

    # Collect all keys and sort: length asc, bsz asc, rank asc
    def sort_key(k: RecordKey):
        length, bsz, rank = k
        return (length, bsz, rank)
    all_keys = sorted(merged.keys(), key=sort_key)

    for length, bsz, rank in all_keys:
        vals = merged[(length, bsz, rank)]
        def g(method: str, metric: str) -> str:
            d = vals.get(method)
            if not d:
                return ""
            if d.get("oom"):
                return "OOM"
            v = d.get(metric)
            if v is None:
                return ""
            return f"{v:.2f}"

        row: Dict[str, str] = {
            "length": str(length),
            "bsz": str(bsz),
            "rank": "" if rank is None else str(rank),
            # Memory (GB)
            "Baseline (GB)": g("Baseline", "mem"),
            "ShadowKV (GB)": g("ShadowKV", "mem"),
            "xKey-1 (GB)": g("xKey-1", "mem"),
            "xKey-2 (GB)": g("xKey-2", "mem"),
            "xKey-4 (GB)": g("xKey-4", "mem"),
            # Throughput (token/s)
            "Baseline (token/s)": g("Baseline", "thr"),
            "ShadowKV (token/s)": g("ShadowKV", "thr"),
            "xKey-1 (token/s)": g("xKey-1", "thr"),
            "xKey-2 (token/s)": g("xKey-2", "thr"),
            "xKey-4 (token/s)": g("xKey-4", "thr"),
        }
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, str]], out_path: str) -> None:
    import csv

    fieldnames = [
        "length",
        "bsz",
        "rank",
        # Memory first (GB)
        "Baseline (GB)",
        "ShadowKV (GB)",
        "xKey-1 (GB)",
        "xKey-2 (GB)",
        "xKey-4 (GB)",
        # Then throughput (token/s)
        "Baseline (token/s)",
        "ShadowKV (token/s)",
        "xKey-1 (token/s)",
        "xKey-2 (token/s)",
        "xKey-4 (token/s)",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate performance logs into CSV (split memory and throughput columns)")
    parser.add_argument("--logs", default="logs", help="Directory containing .log files")
    parser.add_argument("--out", default="e2e.csv", help="Output CSV path")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.logs):
        print(f"Logs directory not found: {args.logs}")
        return 2

    results = parse_logs(args.logs)
    if not results:
        print("No records found in logs")
        return 1

    rows = to_rows(results)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
