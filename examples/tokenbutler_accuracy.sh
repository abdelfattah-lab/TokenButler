#!/bin/bash
# Reproduce TokenButler accuracy results on the RULER benchmark.
#
# Evaluates on 10 datasets: N-S1, N-S2, N-MK1, N-MK2, N-MQ, N-MV, QA-1, QA-2, VT, FWE
# Model: Llama-3.1-8B-Instruct
# Context: 65K tokens, Sparse budget: 2048
#
# Prerequisites:
#   1. Download predictor weights: bash scripts/download_weights.sh
#   2. Build RULER data: bash examples/build_ruler_data.sh
#
# Usage:
#   bash examples/tokenbutler_accuracy.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS="ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/qa_1,ruler/qa_2,ruler/vt,ruler/fwe"

python test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --method KeySifter \
    --datalen 65536 \
    --dataset_name "$DATASETS" \
    --sparse_budget 2048 \
    --chunk_size 8 \
    --rank 160 \
    --predictor_path L3_8Bi_d16_i512_pf4.pt \
    --dDash 16 \
    --producer_frequency 4 \
    --keysifter_intermediate_dim 512

echo ""
echo "Results saved to archive/Meta-Llama-3.1-8B-Instruct/"
