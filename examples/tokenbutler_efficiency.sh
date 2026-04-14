#!/bin/bash
# Reproduce TokenButler efficiency benchmarks.
#
# Measures decode latency (ms/token) with sparse budget 8K for:
#   - GPU: context lengths 32K, 64K, 128K (gen_length=1024)
#   - CPU offloading: context lengths 256K, 512K, 1M (gen_length=128)
#
# Configurations tested:
#   - Dense (full attention)
#   - TokenButler with intervals 1, 2, 4, 8, 16 (+neighbor fetch)
#   - Oracle baselines (random, contiguous)
#
# Prerequisites:
#   1. Download predictor weights: bash scripts/download_weights.sh
#
# Usage:
#   bash examples/tokenbutler_efficiency.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Running efficiency benchmarks..."
echo "This will take several hours. Results are saved incrementally to:"
echo "  test/output/efficiency_budget8K_1M/decoding_time_vs_context.csv"
echo ""

python test/run_missing_configs.py
