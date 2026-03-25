#!/bin/bash
#
# Master script: Full KeySifter accuracy evaluation across all configurations.
#
# Evaluates 3 modes x 5 predict_interval configs x 10 RULER datasets:
#   1. Single-Turn:    context+query prefilled together, then generate
#   2. Multi-Turn (dense cont):  prefill context, prefill_cont query with i=1, then generate with i=N
#   3. Multi-Turn (sparse cont): prefill context, prefill_cont query with i=N, then generate with i=N
#
# Resumable: skips any (mode, config, dataset) combination that already has a result file.
#
# Usage:
#   bash test/run_full_keysifter_eval.sh              # run everything
#   bash test/run_full_keysifter_eval.sh --dry-run     # show what would run
#

set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON=/home/afa55/.conda/envs/xkv_env/bin/python
EVAL_SCRIPT=test/eval_acc.py

# Model and KeySifter params
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
METHOD="KeySifter"
DATALEN=131072
SPARSE_BUDGET=2048
CHUNK_SIZE=8
RANK=160
RANK_K=96
RANK_V=144
GROUP_SIZE=1
DDASH=16
PRODUCER_FREQ=4
INTERMEDIATE_DIM=512
PREDICTOR_PATH="L3_8Bi_d16_i512_pf4.pt"

# Datasets (excluding niah_multiturn which has a different schema)
# TODO: Let's add niah_multiturn results in the results table as well. We can run it separately after collecting results from the others if needed.
DATASETS=(
    "ruler/niah_single_1"
    "ruler/niah_single_2"
    "ruler/niah_multikey_1"
    "ruler/niah_multikey_2"
    "ruler/niah_multiquery"
    "ruler/niah_multivalue"
    "ruler/qa_1"
    "ruler/qa_2"
    "ruler/vt"
    "ruler/fwe"
)

# Configs: name predict_interval enable_neighbor_fetch
CONFIGS=(
    "i1      1  no"
    "i2_nb   2  yes"
    "i4_nb   4  yes"
    "i8_nb   8  yes"
    "i16_nb  16 yes"
)

# Modes: name inference_mode no_prefill_cont_dense
MODES=(
    "single_turn         single_turn  no"
    "multi_turn_dense    multi_turn   no"
    "multi_turn_sparse   multi_turn   yes"
)

# Output directory
OUTPUT_DIR="archive/Meta-Llama-3.1-8B-Instruct/keysifter_full_eval_20260324"
mkdir -p "$OUTPUT_DIR"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Helper: result filename for a given dataset
result_filename() {
    local ds="$1"
    local ds_short="${ds#ruler/}"
    echo "${ds_short}_${DATALEN}_${METHOD}_b${SPARSE_BUDGET}_c${CHUNK_SIZE}_x${GROUP_SIZE}_r${RANK}_k${RANK_K}_v${RANK_V}.jsonl"
}

# Helper: check if a result is complete (file exists and has >= 1 line)
is_complete() {
    local fpath="$1"
    if [[ -f "$fpath" ]] && [[ $(wc -l < "$fpath") -ge 1 ]]; then
        return 0
    fi
    return 1
}

total_runs=0
skipped_runs=0
completed_runs=0
failed_runs=0

for mode_spec in "${MODES[@]}"; do
    read -r mode_name inference_mode no_pcd <<< "$mode_spec"

    for config_spec in "${CONFIGS[@]}"; do
        read -r config_name predict_interval enable_nb <<< "$config_spec"

        # Build config-specific output dir
        cfg_dir="${OUTPUT_DIR}/${mode_name}/${config_name}"
        mkdir -p "$cfg_dir"

        # Check which datasets need running
        datasets_to_run=()
        for ds in "${DATASETS[@]}"; do
            fname=$(result_filename "$ds")
            if is_complete "${cfg_dir}/${fname}"; then
                ((skipped_runs++))
            else
                datasets_to_run+=("$ds")
            fi
            ((total_runs++))
        done

        if [[ ${#datasets_to_run[@]} -eq 0 ]]; then
            echo "[SKIP] ${mode_name}/${config_name}: all ${#DATASETS[@]} datasets complete"
            continue
        fi

        echo ""
        echo "================================================================"
        echo "  Mode: ${mode_name}  Config: ${config_name}  (i=${predict_interval}, nb=${enable_nb})"
        echo "  Datasets to run: ${#datasets_to_run[@]}/${#DATASETS[@]}"
        echo "================================================================"

        if $DRY_RUN; then
            echo "  [DRY RUN] Would run: ${datasets_to_run[*]}"
            continue
        fi

        # Build command
        dataset_str=$(IFS=,; echo "${datasets_to_run[*]}")
        cmd=(
            "$PYTHON" "$EVAL_SCRIPT"
            --model_name "$MODEL_NAME"
            --method "$METHOD"
            --datalen "$DATALEN"
            --dataset_name "$dataset_str"
            --sparse_budget "$SPARSE_BUDGET"
            --chunk_size "$CHUNK_SIZE"
            --rank "$RANK"
            --rank_k "$RANK_K"
            --rank_v "$RANK_V"
            --group_size "$GROUP_SIZE"
            --dDash "$DDASH"
            --producer_frequency "$PRODUCER_FREQ"
            --keysifter_intermediate_dim "$INTERMEDIATE_DIM"
            --predictor_path "$PREDICTOR_PATH"
            --predict_interval "$predict_interval"
            --inference_mode "$inference_mode"
        )
        if [[ "$enable_nb" == "yes" ]]; then
            cmd+=(--enable_neighbor_fetch)
        fi
        if [[ "$no_pcd" == "yes" ]]; then
            cmd+=(--no_prefill_cont_dense)
        fi

        echo "  Running: ${cmd[*]}"
        echo ""

        if "${cmd[@]}"; then
            # Move results from default output path to our structured dir
            model_short="${MODEL_NAME##*/}"
            for ds in "${datasets_to_run[@]}"; do
                fname=$(result_filename "$ds")
                src="archive/${model_short}/${ds}_${DATALEN}_${METHOD}_b${SPARSE_BUDGET}_c${CHUNK_SIZE}_x${GROUP_SIZE}_r${RANK}_k${RANK_K}_v${RANK_V}.jsonl"
                dst="${cfg_dir}/${fname}"
                if [[ -f "$src" ]]; then
                    mv "$src" "$dst"
                    ((completed_runs++))
                else
                    echo "  WARNING: expected result file not found: $src"
                    ((failed_runs++))
                fi
            done
            echo "  [DONE] ${mode_name}/${config_name}"
        else
            echo "  [FAIL] ${mode_name}/${config_name} exited with error"
            ((failed_runs += ${#datasets_to_run[@]}))
        fi
    done
done

echo ""
echo "================================================================"
echo "  EVALUATION COMPLETE"
echo "  Total: ${total_runs}  Skipped: ${skipped_runs}  Completed: ${completed_runs}  Failed: ${failed_runs}"
echo "================================================================"

if $DRY_RUN; then
    echo "[DRY RUN] No results generated."
    exit 0
fi

# ============================================================
# Generate consolidated results.md
# ============================================================
echo ""
echo "Generating results.md..."

RESULTS_MD="${OUTPUT_DIR}/results.md"

$PYTHON - "$OUTPUT_DIR" "${DATASETS[*]}" << 'PYEOF'
import sys, json, os
from pathlib import Path
from collections import OrderedDict

output_dir = Path(sys.argv[1])
datasets = sys.argv[2].split()

configs = OrderedDict([
    ("i1",    {"interval": 1,  "nb": False}),
    ("i2_nb", {"interval": 2,  "nb": True}),
    ("i4_nb", {"interval": 4,  "nb": True}),
    ("i8_nb", {"interval": 8,  "nb": True}),
    ("i16_nb",{"interval": 16, "nb": True}),
])

modes = OrderedDict([
    ("single_turn",       "Single-Turn"),
    ("multi_turn_dense",  "Multi-Turn (dense cont)"),
    ("multi_turn_sparse", "Multi-Turn (sparse cont)"),
])

# Short display names for datasets
ds_display = {
    "ruler/niah_single_1":  "N-S1",
    "ruler/niah_single_2":  "N-S2",
    "ruler/niah_multikey_1":"N-MK1",
    "ruler/niah_multikey_2":"N-MK2",
    "ruler/niah_multiquery":"N-MQ",
    "ruler/niah_multivalue":"N-MV",
    "ruler/qa_1":           "QA-1",
    "ruler/qa_2":           "QA-2",
    "ruler/vt":             "VT",
    "ruler/fwe":            "FWE",
}

def result_filename(ds):
    ds_short = ds.replace("ruler/", "")
    return f"{ds_short}_131072_KeySifter_b2048_c8_x1_r160_k96_v144.jsonl"

def read_score(fpath):
    if not fpath.exists():
        return None
    lines = [json.loads(l) for l in open(fpath)]
    if not lines:
        return None
    return lines[-1].get("avg_score", None)

with open(output_dir / "results.md", "w") as f:
    f.write("# KeySifter Full Accuracy Evaluation\n\n")
    f.write("## Setup\n\n")
    f.write("- **Model:** meta-llama/Meta-Llama-3.1-8B-Instruct\n")
    f.write("- **Context length:** 131072 (128K)\n")
    f.write("- **Sparse budget:** 2048 tokens\n")
    f.write("- **Predictor:** dDash=16, producer_frequency=4, intermediate_dim=512\n")
    f.write("- **Samples per dataset:** 96\n\n")

    f.write("## Evaluation Modes\n\n")
    f.write("| Mode | Description |\n")
    f.write("|------|-------------|\n")
    f.write("| **Single-Turn** | Context + query are concatenated and prefilled together (dense), then answer is generated with predict_interval=N |\n")
    f.write("| **Multi-Turn (dense cont)** | Context is prefilled (dense). Query is processed via `prefill_cont` with predict_interval forced to **1** (prediction at every step). Answer is generated with predict_interval=N |\n")
    f.write("| **Multi-Turn (sparse cont)** | Context is prefilled (dense). Query is processed via `prefill_cont` with the **same predict_interval=N** as answer generation. Answer is generated with predict_interval=N |\n\n")

    f.write("## Configurations\n\n")
    f.write("| Config | predict_interval (N) | Neighbor Fetch | Description |\n")
    f.write("|--------|---------------------|----------------|-------------|\n")
    for name, cfg in configs.items():
        nb = "Yes (2x sparse buffer)" if cfg["nb"] else "No"
        f.write(f"| {name} | {cfg['interval']} | {nb} | Predict every {cfg['interval']} decode step(s)")
        if cfg["nb"]:
            f.write(" + fetch adjacent tokens")
        f.write(" |\n")
    f.write("\n")

    f.write("## Dataset Abbreviations\n\n")
    f.write("| Abbreviation | Full Name |\n")
    f.write("|-------------|----------|\n")
    for ds, short in ds_display.items():
        f.write(f"| {short} | {ds.replace('ruler/', '')} |\n")
    f.write("\n")

    # Generate a table per mode
    for mode_key, mode_label in modes.items():
        f.write(f"## Results: {mode_label}\n\n")

        # Collect all scores
        all_scores = {}  # config -> ds -> score
        for cfg_name in configs:
            all_scores[cfg_name] = {}
            for ds in datasets:
                fpath = output_dir / mode_key / cfg_name / result_filename(ds)
                all_scores[cfg_name][ds] = read_score(fpath)

        # Config labels
        cfg_labels = []
        for name, cfg in configs.items():
            label = f"i={cfg['interval']}"
            if cfg["nb"]:
                label += "+nb"
            cfg_labels.append(label)

        # Table header
        col_w = 10
        header = f"| {'Dataset':<8} |"
        for label in cfg_labels:
            header += f" {label:>{col_w}} |"
        f.write(header + "\n")
        f.write("|" + "-" * 10 + "|" + (("-" * (col_w + 2) + "|") * len(configs)) + "\n")

        # Data rows
        cfg_avgs = {name: [] for name in configs}
        for ds in datasets:
            short = ds_display.get(ds, ds.replace("ruler/", ""))
            row = f"| {short:<8} |"
            for cfg_name in configs:
                score = all_scores[cfg_name].get(ds)
                if score is not None:
                    row += f" {score:>{col_w}.4f} |"
                    cfg_avgs[cfg_name].append(score)
                else:
                    row += f" {'—':>{col_w}} |"
            f.write(row + "\n")

        # Average row
        f.write("|" + "-" * 10 + "|" + (("-" * (col_w + 2) + "|") * len(configs)) + "\n")
        row = f"| {'AVG':<8} |"
        for cfg_name in configs:
            avgs = cfg_avgs[cfg_name]
            if avgs:
                row += f" {sum(avgs)/len(avgs):>{col_w}.4f} |"
            else:
                row += f" {'—':>{col_w}} |"
        f.write(row + "\n")
        f.write("\n")

print(f"Results written to {output_dir / 'results.md'}")
PYEOF

echo "All done. Results at: ${OUTPUT_DIR}/results.md"
