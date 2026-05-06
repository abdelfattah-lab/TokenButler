#!/bin/bash
# Reproduce all TokenButler experiments and paper plots in one go.
#
# Stages:
#   1. Build RULER benchmark data (skipped if already present).
#   2. Run RULER accuracy sweep (10 datasets, 64K context, sparse_budget=8192).
#   3. Run RULER accuracy sweep with prediction intervals 1/2/4/8/16 (b=8192).
#   4. Run decoding efficiency sweep (32K..1M context, 9 configurations).
#   5. Run detailed timing-breakdown benchmark.
#   6. Render paper plots into paper_plots/.
#
# Prerequisites:
#   - bash scripts/download_weights.sh   (predictor weights at L3_8Bi_d16_i512_pf4.pt)
#   - working .venv (uv sync)
#   - HF cache containing meta-llama/Meta-Llama-3.1-8B-Instruct
#
# Usage:
#   bash examples/reproduce_all.sh           # full reproduction (multi-day on a single A100/A6000)
#   bash examples/reproduce_all.sh --quick   # smaller sweeps for plumbing verification
#
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=""
if [[ "${1:-}" == "--quick" ]]; then
    QUICK="--quick"
    echo "*** QUICK MODE: smaller sweeps, results NOT paper-quality ***"
fi

PLOT_DIR=paper_plots
mkdir -p "$PLOT_DIR"

# Snapshot any pre-existing efficiency CSV so we don't mix legacy rows in
EFFICIENCY_CSV=test/output/efficiency_budget8K_1M/decoding_time_vs_context.csv
if [ -f "$EFFICIENCY_CSV" ] && grep -q "KeySifter" "$EFFICIENCY_CSV"; then
    cp "$EFFICIENCY_CSV" "${EFFICIENCY_CSV%.csv}.legacy_keysifter.csv"
    rm "$EFFICIENCY_CSV"
    echo "Moved legacy KeySifter-labeled CSV aside."
fi

echo ""
echo "===== Stage 1/6: build RULER data ====="
if [ ! -d data/ruler/data/llama-3 ]; then
    bash examples/build_ruler_data.sh
else
    echo "RULER data already present, skipping."
fi

echo ""
echo "===== Stage 2/6: RULER accuracy (b=8192) ====="
bash examples/tokenbutler_accuracy.sh

echo ""
echo "===== Stage 3/6: RULER accuracy with prediction intervals (b=8192) ====="
bash examples/tokenbutler_accuracy_intervals.sh

echo ""
echo "===== Stage 4/6: decoding efficiency vs context length ====="
bash examples/tokenbutler_efficiency.sh

echo ""
echo "===== Stage 5/6: detailed timing-breakdown benchmark ====="
.venv/bin/python test/benchmark_combined_figure.py $QUICK \
    --predictor-path L3_8Bi_d16_i512_pf4.pt \
    --output-dir "$PLOT_DIR"

echo ""
echo "===== Stage 6/6: render plots ====="
.venv/bin/python test/plot_decoding_efficiency.py \
    --csv "$EFFICIENCY_CSV" \
    --output-dir "$PLOT_DIR"
if [ -f "$PLOT_DIR/combined_timing.csv" ]; then
    .venv/bin/python test/plot_timing_breakdown.py \
        --csv "$PLOT_DIR/combined_timing.csv" \
        --output "$PLOT_DIR/timing_breakdown.pdf"
fi

echo ""
echo "===== Done ====="
echo "  Accuracy results :  archive/Meta-Llama-3.1-8B-Instruct/ruler/*.jsonl"
echo "  Efficiency CSV   :  $EFFICIENCY_CSV"
echo "  Combined CSV     :  $PLOT_DIR/combined_timing.csv"
echo "  Plots            :  $PLOT_DIR/"
