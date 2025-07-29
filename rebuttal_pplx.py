import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import matplotlib.pyplot as plt
import matplotlib

# Fixed order for workloads
ordered_workloads = ["TokenButler", "H2O", "Quest", "SnapKV", "StreamingLLM", "Oracle"]

# Fixed colors (you can tweak as needed)
workload_colors = {
    "TokenButler": "#1f77b4",   # Blue
    "H2O": "#2ca02c",           # Green
    "Quest": "#d62728",         # Red
    "SnapKV": "#9467bd",        # Purple
    "StreamingLLM": "#8c564b",  # Brown
    "Oracle": "#ff7f0e",        # Orange
}
# ------------------------------------------------------------------
# 1.  File → workload-name mapping  (updated)
# ------------------------------------------------------------------
name_mapping = {
    "_Rebuttal_ExpPred.csv": "TokenButler",   # new “TokenButler”
    "_Rebuttal_Oracle.csv":   "Oracle",       # new “Oracle”
    "_h2o_true.csv":          "H2O",
    "_quest.csv":             "Quest",
    "_snapkv.csv":            "SnapKV",
    "_streamingLLM.csv":      "StreamingLLM",
}

# ------------------------------------------------------------------
# 2.  Model-size → pretty label mapping
#     (derived from filename.split('_')[1])
# ------------------------------------------------------------------
model_name_map = {
    "1B": "Llama-3.2-1B",
    "3B": "Llama-3.2-3B",
    "8B": "Llama-3.1-8B",
}

# ------------------------------------------------------------------
# 3.  Data ingestion
# ------------------------------------------------------------------
def read_all_results(base_path: str):
    """
    Scan `base_path` for all CSVs that match any suffix in `name_mapping`,
    combine them per model size, and return a dict keyed by model label.
    """
    combined = {}  # {model_label: dataframe}

    for fname in os.listdir(base_path):
        # Only process files that end with one of our known suffixes
        suffix = next((s for s in name_mapping.keys() if fname.endswith(s)), None)
        if suffix is None:
            continue

        # Extract model-size token (1B / 3B / 8B) from 2nd field
        size_token = fname.split("_")[1]  # e.g. "1B"
        model_label = model_name_map.get(size_token)
        if model_label is None:
            continue  # skip unknown sizes just in case

        # Load, deduplicate header rows exactly as before
        fpath = os.path.join(base_path, fname)
        with open(fpath, "r") as f:
            lines = f.readlines()

        filtered = []
        header_seen = False
        for line in lines:
            if line.startswith("seed,model_path"):
                if not header_seen:
                    filtered.append(line)
                    header_seen = True
            else:
                filtered.append(line)

        df = pd.read_csv(io.StringIO("".join(filtered)))
        df["wname"] = name_mapping[suffix]

        # Ensure correct type for perplexity (and sparsity just in case)
        if "perplexity" in df.columns:
            df["perplexity"] = pd.to_numeric(df["perplexity"], errors="coerce")
        if "true_token_sparsity" in df.columns:
            df["true_token_sparsity"] = pd.to_numeric(df["true_token_sparsity"], errors="coerce")
        # Stash
        combined.setdefault(model_label, []).append(df)

    # Concatenate lists into single DF per model
    for model_label, dfs in combined.items():
        combined[model_label] = pd.concat(dfs, ignore_index=True)

    return combined


# ------------------------------------------------------------------
# 4.  Plotting (perplexity only, 1 row × 3 cols)
# ------------------------------------------------------------------
def plot_perplexity(models_data: dict, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    rows, cols = 1, 3
    fig, axes = plt.subplots(rows, cols, figsize=(24, 6), sharey=False)
    if cols == 1:  # safety for future
        axes = [axes]

    for ax, (model_label, df) in zip(axes, models_data.items()):
        for wname in ordered_workloads:
            if wname not in df['wname'].values:
                continue
            sub = df[df["wname"] == wname].sort_values("true_token_sparsity")
            ax.plot(
                sub["true_token_sparsity"],
                sub["perplexity"],
                label=wname,
                marker="o",
                linestyle="-",
                linewidth=2.5,
                color=workload_colors[wname],
            )

        # Style tweaks identical to your reference
        ax.set_title(model_label, fontsize=26)
        ax.set_xlabel("Net Token Sparsity (%)", fontsize=26)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis="both", labelsize=26)

        # y-limits using new Oracle / H2O rows
        min_ppl = df[df["wname"] == "Oracle"]["perplexity"].min()
        max_ppl = df[df["wname"] == "H2O"]["perplexity"].max()
        # ax.set_ylim(min_ppl * 0.99, max_ppl * 1.3)
        ax.set_ylim(min_ppl * 0.99, max_ppl * 1.04)

    # Hide any unused axes (if fewer than 3 models found)
    for ax in axes[len(models_data) :]:
        ax.axis("off")

    axes[0].set_ylabel("Perplexity", fontsize=26)
    # Explicit, fixed legend
    legend_handles = [
        matplotlib.lines.Line2D([0], [0], color=workload_colors[w], marker='o', label=w, linewidth=2.5)
        for w in ordered_workloads if any(df['wname'].eq(w).any() for df in models_data.values())
    ]

    fig.legend(
        handles=legend_handles,
        loc='upper center',
        fontsize=26,
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, 1.03)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = os.path.join(output_dir, "perplexity_comparison.pdf")
    print(f"Saving plot to {out_path}")
    plt.savefig(out_path)
    plt.close()

import numpy as np                       # ← add to your imports

# ── 6.  Markdown table builder ─────────────────────────────────────────
def markdown_tables_from_sparsity(models_data: dict,
                                  nan_placeholder: str = "—"):
    """
    For each model in `models_data` (as returned by read_all_results)
    print a Markdown table:

        • Rows  …… workloads (TokenButler, H2O, …)
        • Cols  …… aligned position in sparsity‑rank order
        • Header… mean sparsity of all non‑missing rows in that column
        • Cell   … perplexity (two decimals) or placeholder if absent
    """
    for model_label, df in models_data.items():
        # 1) collect each workload’s rows sorted by sparsity
        series_by_w = {}
        max_len = 0
        for w in ordered_workloads:
            sub = (
                df[df["wname"] == w]
                .sort_values("true_token_sparsity")
                [["true_token_sparsity", "perplexity"]]
                .reset_index(drop=True)
            )
            series_by_w[w] = sub
            max_len = max(max_len, len(sub))

        # # 2) compute column headers = mean sparsity at each rank
        # avg_sparsities = []
        # for idx in range(max_len):
        #     sparsities_here = [
        #         series_by_w[w].loc[idx, "true_token_sparsity"]
        #         for w in ordered_workloads
        #         if idx < len(series_by_w[w])
        #     ]
        #     avg_sparsities.append(np.nanmean(sparsities_here))

        # 2) keep only column‑indices where at least one workload has a
        #    *non‑NaN* perplexity value
        indices_to_keep = [
            idx for idx in range(max_len)
            if any(
                idx < len(series_by_w[w])
                and not pd.isna(series_by_w[w].loc[idx, "perplexity"])
                for w in ordered_workloads
            )
        ]

        # 3) compute column headers = mean sparsity of the kept indices
        avg_sparsities = []
        for idx in indices_to_keep:
            sparsities_here = [
                series_by_w[w].loc[idx, "true_token_sparsity"]
                for w in ordered_workloads
                if idx < len(series_by_w[w])
            ]
            avg_sparsities.append(np.nanmean(sparsities_here))
        # Pre‑compute best (lowest) perplexity per column, **excluding Oracle**
        best_by_col = []
        for idx in indices_to_keep:
            vals = [
                series_by_w[w].loc[idx, "perplexity"]
                for w in ordered_workloads
                if w != "Oracle" and idx < len(series_by_w[w])
            ]
            best_by_col.append(min(vals) if vals else np.nan)

        col_headers = [f"{s:.1f}%" for s in avg_sparsities]

        # 3) build the Markdown table rows
        header     = ["Workload"] + col_headers
        separator  = ["-" * len(h) for h in header]
        table_rows = [header, separator]

        for w in ordered_workloads:
            row = [w]
            sub = series_by_w[w]
            for idx in indices_to_keep:
                if idx < len(sub):
                    # row.append(f"{sub.loc[idx, 'perplexity']:.2f}")
                    val = sub.loc[idx, "perplexity"]
                    cell = f"{val:.2f}"
                    # Bold if this workload is NOT Oracle and ties for best in the column
                    if w != "Oracle" and np.isclose(val, best_by_col[idx]):
                        cell = f"**{cell}**"
                    row.append(cell)
                else:
                    row.append(nan_placeholder)
            table_rows.append(row)

        # 4) pretty‑print
        print(f"\n**{model_label}**\n")
        for r in table_rows:
            print(" | ".join(r))
        print()  # blank line between models


# ── 7.  Main (updated) ────────────────────────────────────────────────
if __name__ == "__main__":
    base = "./evalresultspplx"
    data_by_model = read_all_results(base)

    # Comment this out if you no longer need the PDF plots.
    # plot_perplexity(data_by_model)

    # Print the three Markdown tables (one for each model)
    markdown_tables_from_sparsity(data_by_model)

# # ------------------------------------------------------------------
# # 5.  Main
# # ------------------------------------------------------------------
# if __name__ == "__main__":
#     base = "./evalresultspplx"
#     data_by_model = read_all_results(base)
#     plot_perplexity(data_by_model)

    # # Global legend (top center)
    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(
    #     handles,
    #     labels,
    #     ncol=len(labels),
    #     fontsize=26,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.03),
    # )