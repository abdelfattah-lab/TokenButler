#!/bin/bash
# Reproduce TokenButler accuracy with prediction intervals.
#
# Tests prediction step intervals: 1, 2, 4, 8, 16
# Intervals > 1 use neighbor fetching for improved accuracy.
#
# Model: Llama-3.1-8B-Instruct
# Context: 65K tokens, Sparse budget: 8192
#
# Prerequisites:
#   1. Download predictor weights: bash scripts/download_weights.sh
#   2. Build RULER data: bash examples/build_ruler_data.sh
#
# Usage:
#   bash examples/tokenbutler_accuracy_intervals.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS="ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/qa_1,ruler/qa_2,ruler/vt,ruler/fwe"

COMMON_ARGS="--model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --method TokenButler \
    --datalen 65536 \
    --dataset_name $DATASETS \
    --sparse_budget 8192 \
    --chunk_size 8 \
    --rank 160 \
    --predictor_path L3_8Bi_d16_i512_pf4.pt \
    --dDash 16 \
    --producer_frequency 4 \
    --tokenbutler_intermediate_dim 512"

echo "=== Interval 1 (baseline, no neighbor fetch) ==="
python test/eval_acc.py $COMMON_ARGS --predict_interval 1

echo ""
echo "=== Interval 2 + Neighbor Fetch ==="
python test/eval_acc.py $COMMON_ARGS --predict_interval 2 --enable_neighbor_fetch

echo ""
echo "=== Interval 4 + Neighbor Fetch ==="
python test/eval_acc.py $COMMON_ARGS --predict_interval 4 --enable_neighbor_fetch

echo ""
echo "=== Interval 8 + Neighbor Fetch ==="
python test/eval_acc.py $COMMON_ARGS --predict_interval 8 --enable_neighbor_fetch

echo ""
echo "=== Interval 16 + Neighbor Fetch ==="
python test/eval_acc.py $COMMON_ARGS --predict_interval 16 --enable_neighbor_fetch

echo ""
echo "Results saved to archive/Meta-Llama-3.1-8B-Instruct/"
