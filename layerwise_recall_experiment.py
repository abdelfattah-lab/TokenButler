#!/usr/bin/env python3
"""
Summarise recall@K for layers 25‑27.

Reads the debug‑hitrates CSV (default path: L3_1B_PL24_debug_hitrates.csv),
averages top‑10 / top‑30 / top‑50 recall across *all* heads for the
requested layers, and prints a ready‑to‑paste Markdown table.
"""

import sys
import pandas as pd

def main(csv_path: str = "L3_3B_PL24_debug_hitrates.csv") -> None:
    # Columns we need
    use_cols = ["layer_idx", "top10", "top30", "top50"]

    # Load and keep only rows for layers 25‑27
    df = (
        pd.read_csv(csv_path, usecols=use_cols)
        .query("layer_idx in [25, 26, 27]")
        .groupby("layer_idx", sort=True)[["top10", "top30", "top50"]]
        .mean()
        .round(4)   # round for neat printing
    )

    # Build & print the Markdown table
    header = "| layer_idx | top10 | top30 | top50 |\n|-----------|-------|-------|-------|"
    rows   = "\n".join(
        f"| {idx} | {row.top10:.4f} | {row.top30:.4f} | {row.top50:.4f} |"
        for idx, row in df.iterrows()
    )
    print(f"{header}\n{rows}")

if __name__ == "__main__":
    # Optional CLI: python script.py [path/to/csv]
    main(sys.argv[1] if len(sys.argv) > 1 else "L3_3B_PL0_debug_hitrates.csv")
