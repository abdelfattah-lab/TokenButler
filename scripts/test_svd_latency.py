#!/usr/bin/env python3
"""
Benchmark SVD latency for a matrix of shape [length, dim * gs] on CUDA only.

Tests:
- torch.linalg.svd (full_matrices=False)
- torch.svd_lowrank (q=rank)

Dtypes: fp32 only.

Usage:
    python scripts/test_svd_latency.py --length 65536 131072 --gs 1 2 4 --rank 64 128 --dim 1024
"""

import argparse
import time
import torch
import csv
import os


def _time_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / 10 * 1000.0


def benchmark(length: int, dim: int, gs: int, rank: int):
    rows = length
    cols = dim * gs
    rank *= gs

    # Create random matrix (fp32 on CUDA)
    g = torch.Generator(device="cuda").manual_seed(0)
    A = torch.randn(4, rows, cols, device="cuda", dtype=torch.float32, generator=g)
    r = min(rank, min(rows, cols))

    methods = [
        ("linalg.svd", lambda: torch.linalg.svd(A, full_matrices=False)),
        ("svd_lowrank", lambda: torch.svd_lowrank(A, q=r)),
    ]

    times = []
    for _, fn in methods:
        fn()  # warmup
        t = _time_ms(fn)
        times.append(float(t))

    del A
    torch.cuda.empty_cache()

    return times[0], times[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark SVD latency [length, dim*gs] on CUDA")
    p.add_argument("--length", type=int, nargs="+", default=[32768, 65536, 131072, 262144], help="List of sequence lengths (rows)")
    p.add_argument("--gs", type=int, nargs="+", default=[1, 2, 4], help="List of group sizes (cols = dim*gs)")
    p.add_argument("--rank", type=int, nargs="+", default=[64, 96, 128, 160], help="List of target ranks for torch.svd_lowrank (q)")
    p.add_argument("--dim", type=int, default=1024, help="Hidden dimension per layer (num_head * head_dim)")
    return p.parse_args()


def main():
    args = parse_args()
    with open("svd_latency.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "m",
            "n",
            "gs",
            "rank",
            "torch.linalg.svd",
            "torch.svd_lowrank",
            "avg_linalg_per_layer",
            "avg_svd_lowrank_per_layer",
        ])
        for length in args.length:
            for rank in args.rank:
                for gs in args.gs:
                    t_linalg, t_lowrank = benchmark(length=int(length), dim=int(args.dim), gs=int(gs), rank=int(rank))
                    n = int(args.dim) * int(gs)
                    rank = int(rank) * int(gs)
                    avg_linalg = t_linalg / float(gs)
                    avg_lowrank = t_lowrank / float(gs)

                    print(f"Length: {length}, Dim: {n}, GS: {gs}, Rank: {rank} => avg_linalg: {avg_linalg:.2f}, avg_lowrank: {avg_lowrank:.2f}")

                    writer.writerow([
                        length,
                        n,
                        gs,
                        rank,
                        f"{t_linalg:.2f}",
                        f"{t_lowrank:.2f}",
                        f"{avg_linalg:.2f}",
                        f"{avg_lowrank:.2f}",
                    ])
                    f.flush()
                    os.fsync(f.fileno())


if __name__ == "__main__":
    main()

