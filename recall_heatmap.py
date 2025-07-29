#!/usr/bin/env python3
"""
multi_model_recall.py

Read all *_debug_hitrates.csv files in the current directory and plot a
multi‑model mean‑recall curve with ±1 σ shading for each model.

Output → multi_model_recall.pdf
"""

import glob
import os
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# ── 1. locate CSVs ──────────────────────────────────────────────────────
csv_paths = sorted(glob.glob("*_debug_hitrates.csv"))
if not csv_paths:
    raise FileNotFoundError("No *_debug_hitrates.csv files found in the cwd.")

# Mapping filename → legend label
NAME_MAP = {
    "l3_8b_debug_hitrates.csv": "Llama‑3.1‑8B",
    "l3_3b_debug_hitrates.csv": "Llama‑3.2‑3B",
    "l3_3b_inst_debug_hitrates.csv": "Llama‑3.2‑3B-Instruct",
    "l3_1b_debug_hitrates.csv": "Llama‑3.2‑1B",
}

# Ensure deterministic colour picking (tab10 has 10 distinguishable colours)
cmap = get_cmap("tab10")
colour_cycle = [cmap(i) for i in range(len(csv_paths))]

# Make fonts larger for everything
plt.rcParams.update({
    "font.size":       18,
    "axes.titlesize":  18,
    "axes.labelsize":  18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

# ── 2. gather data ──────────────────────────────────────────────────────
model_curves = OrderedDict()   # filename → (Ks, mean, std)

for path in csv_paths:
    df = pd.read_csv(path)

    # find the top‑K columns (top1, top2, …)
    top_cols = sorted(
        (c for c in df.columns if re.fullmatch(r"top\d+", c)),
        key=lambda c: int(c[3:])
    )
    if not top_cols:
        raise RuntimeError(f"No top‑K columns found in {path}.")

    ks     = np.array([int(c[3:]) for c in top_cols])
    means  = df[top_cols].mean().values
    stds   = df[top_cols].std().values

    model_curves[path] = (ks, means, stds)

# ── 3. plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for (idx, (path, (ks, means, stds))) in enumerate(model_curves.items()):
    color = colour_cycle[idx]

    label = NAME_MAP.get(os.path.basename(path), os.path.basename(path))

    # shaded ±1 σ region
    ax.fill_between(
        ks,
        means - stds,
        means + stds,
        color=color,
        alpha=0.20,
    )
    ax.plot(
        ks, means,
        label=label,
        color=color,
        linewidth=2.5,
    )
    ax.scatter(ks, means, color=color, edgecolor='k', zorder=3)
# ── 4. cosmetics ────────────────────────────────────────────────────────
ax.set_xlabel("K in top‑K (%)")
ax.set_ylabel("Recall")
ax.set_title("Mean recall across all heads & layers")
ax.set_xlim(min(ks), max(ks))
ax.set_ylim(0.3, 1)
ax.set_xticks([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90])
# ax.set_xscale("log")
ax.grid(ls="--", lw=0.4, alpha=0.6)
ax.legend(loc="lower right", frameon=False)

plt.tight_layout()
fig.savefig("multi_model_recall.pdf")
plt.close(fig)
# ── 3. build Markdown tables instead of a figure ───────────────────────
from textwrap import indent

# Collect the (sorted) union of all K values so every model shares a header
all_ks = sorted({k for ks, _, _ in model_curves.values() for k in ks})

# Helper to build a table (pass in either means or stds)
def build_table(value_idx: int, title: str) -> str:
    header = ["Model"] + [str(k) for k in all_ks]
    rows   = [header, ["-"*len(h) for h in header]]  # separator row for Markdown

    for path, (ks, means, stds) in model_curves.items():
        name   = NAME_MAP.get(os.path.basename(path), os.path.basename(path))
        values = means if value_idx == 0 else stds
        # Map KS→value, then emit in the global‑K order
        kv_map = {k: v for k, v in zip(ks, values)}
        row = [name] + [f"{kv_map[k]:.2f}" if k in kv_map else "—" for k in all_ks]
        rows.append(row)

    # Convert to Markdown
    md_lines = [" | ".join(r) for r in rows]
    return f"**{title}**\n\n" + "\n".join(md_lines) + "\n"

# Build the tables
mean_table = build_table(0, "Mean recall across all heads & layers")
std_table  = build_table(1, "Standard deviation (σ)")

# Print them so they can be copy‑pasted into the rebuttal
print(mean_table)
print(std_table)

import pdb; pdb.set_trace()
print("✓  Plot written to multi_model_recall.pdf")

#!/usr/bin/env python3
"""
Draw per‑head, per‑layer recall heat‑maps from debug_hitrates.csv.
Each metric (top1 … top50) → one file  recall_heatmap/<metric>.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASEMODEL = "l3_3b_inst"
CSV_PATH = f"{BASEMODEL}_debug_hitrates.csv"
OUT_DIR  = f"{BASEMODEL}_recall_heatmap"
OUT_DIR  = f"{BASEMODEL}_recall_heatmap"
OUT_FIG  = f"{BASEMODEL}_recall_curve.png"
# ── I/O ──────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
os.makedirs(OUT_DIR, exist_ok=True)

# assure integer indices
df["layer_idx"] = df["layer_idx"].astype(int)
df["head_idx"]  = df["head_idx"].astype(int)

# dynamic ranges ───────────────────────────────────────
head_range  = range(df["head_idx"].min(),  df["head_idx"].max()  + 1)
layer_range = range(df["layer_idx"].min(), df["layer_idx"].max() + 1)
metrics = [c for c in df.columns if c.startswith("top")]

# ── heat‑map per metric ─────────────────────────────────────────────────
for metric in metrics:
    # pivot → rows=head 0‑31, cols=layer 1‑31, values=mean recall
    mat = (
        df.groupby(["head_idx", "layer_idx"])[metric]
          .mean()
          .unstack(fill_value=np.nan)
          .reindex(index=head_range)         # use data‑driven head range
          .reindex(columns=layer_range)      # use data‑driven layer range
          .to_numpy()
    )

    plt.figure(figsize=(12, 6))
    img = plt.imshow(mat, aspect="auto", origin="lower")
    plt.colorbar(img, label=metric)
    plt.title(f"{metric} recall heat‑map")
    plt.xlabel("Layer index")
    plt.ylabel("Head index")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{metric}.png"))
    plt.close()

metric = "top1"
# ── new: sort (head, layer) pairs by recall for current metric ──
recall_values = (
    df.groupby(["head_idx", "layer_idx"])[metric]
    .mean()
    .reset_index()
)

sorted_pairs = sorted(
    zip(recall_values["head_idx"], recall_values["layer_idx"], recall_values[metric]),
    key=lambda x: x[2]  # sort by recall
)

lowest_pairs = [ (h, l) for h, l, _ in sorted_pairs ]

print(f"\n# Lowest {metric} recall positions:")
print(lowest_pairs)

print(f"✓ Heat‑maps written to {OUT_DIR}/")
exit(0)

#!/usr/bin/env python3
"""
Draw a single mean‑recall curve (±1 σ shaded) for every top‑K column
found in debug_hitrates.csv, colouring the line with a viridis gradient.

Output → recall_heatmap/recall_curve.png
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, collections as mcoll



# ── 1. load & identify the top‑K columns ────────────────────────────────
df = pd.read_csv(CSV_PATH)
top_cols = [c for c in df.columns if re.fullmatch(r"top\d+", c)]
if not top_cols:
    raise RuntimeError("No columns named like 'topX' found in CSV.")

# sort numerically by the K value (e.g. top1, top2, …)
top_cols = sorted(top_cols, key=lambda c: int(c[3:]))

# ── 2. aggregate: mean & std across everything ──────────────────────────
means = df[top_cols].mean()
stds  = df[top_cols].std()

# x‑axis: the numeric Ks (1,2,3…50)
x_vals = [int(c[3:]) for c in top_cols]

# ── 3. prepare gradient colours along the line ──────────────────────────
cmap  = cm.get_cmap("viridis")
norm  = plt.Normalize(min(x_vals), max(x_vals))
colors = cmap(norm(x_vals))  # RGBA for each point

# helper: split line into coloured segments
def line_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)

segments = line_segments(x_vals, means.values)
lc = mcoll.LineCollection(segments, colors=colors[:-1], linewidth=2)

# ── 4. plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

# shaded error region
ax.fill_between(
    x_vals,
    means - stds,
    means + stds,
    color='grey',
    alpha=0.25,
    label="±1 σ"
)

# coloured mean line
ax.add_collection(lc)
ax.scatter(x_vals, means, c=colors, edgecolor='k', zorder=3)  # one dot per K

# cosmetics
ax.set_xlim(min(x_vals), max(x_vals))
ax.set_ylim(0, 1)
ax.set_xticks(x_vals)
ax.set_xlabel("K in top‑K (%)")
ax.set_ylabel("Recall")
ax.set_title("Mean recall across all heads & layers\n(shaded = ±1 σ)")
ax.grid(True, ls="--", lw=0.4, alpha=0.6)
ax.legend()

# colour‑bar for reference
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("K (%)")

# ── 5. save ─────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, OUT_FIG))
plt.close(fig)

print(f"✓  Plot written to {os.path.join(OUT_DIR, OUT_FIG)}")
