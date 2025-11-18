import pandas as pd
import matplotlib.pyplot as plt

# --- Load and clean data ---
csv_path = "./evalresults/L3_8B_Eval.csv"

# Read CSV (headers are on every other line)
df = pd.read_csv(csv_path)

# Drop fully empty rows
df = df.dropna(how="all")

# Drop the repeated header rows (where first column is literally the header name)
if "proj_name" in df.columns:
    df = df[df["proj_name"] != "proj_name"]

# Keep only the two eval modes of interest and map to display names
mode_map = {
    "exppred": "TokenButler",
    "oracle": "Oracle",
}

df["eval_llm_mode"] = df["eval_llm_mode"].astype(str).str.strip()
df["mode_label"] = df["eval_llm_mode"].str.lower().map(mode_map)
df = df[df["mode_label"].notna()].copy()

# Identify all *_acc columns
acc_cols = [c for c in df.columns if c.endswith("_acc")]

# Convert numeric columns to proper dtypes
num_cols = ["true_token_sparsity", "perplexity"] + acc_cols
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows missing critical values
df = df.dropna(subset=["true_token_sparsity", "perplexity"])

# Compute average accuracy across all *_acc columns
df["avg_acc"] = df[acc_cols].mean(axis=1, skipna=True)

# Aggregate by (mode_label, true_token_sparsity) to get smooth curves
agg = (
    df.groupby(["mode_label", "true_token_sparsity"], as_index=False)
      .agg({"perplexity": "mean", "avg_acc": "mean"})
      .sort_values(["mode_label", "true_token_sparsity"])
)

# --- Plotting ---
plt.figure(figsize=(10, 4))
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

# Left plot: perplexity vs true_token_sparsity
ax0 = axes[0]
for mode_name in ["TokenButler", "Oracle"]:
    sub = agg[agg["mode_label"] == mode_name]
    if not sub.empty:
        ax0.plot(
            sub["true_token_sparsity"],
            sub["perplexity"],
            marker="o",
            label=mode_name,
        )

ax0.set_xlabel("True Token Sparsity")
ax0.set_ylabel("Perplexity")
ax0.set_title("Perplexity vs True Token Sparsity")
ax0.grid(True, linestyle="--", alpha=0.4)
ax0.legend()

# Right plot: avg accuracy vs true_token_sparsity
ax1 = axes[1]
for mode_name in ["TokenButler", "Oracle"]:
    sub = agg[agg["mode_label"] == mode_name]
    if not sub.empty:
        ax1.plot(
            sub["true_token_sparsity"],
            sub["avg_acc"],
            marker="o",
            label=mode_name,
        )

ax1.set_xlabel("True Token Sparsity")
ax1.set_ylabel("Average Accuracy")
ax1.set_title("Average Accuracy vs True Token Sparsity")
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.legend()

plt.tight_layout()
plt.savefig("8b_perf.pdf")
plt.close(fig)
