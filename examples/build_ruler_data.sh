#!/bin/bash
# Build RULER benchmark datasets for Llama-3.1-8B-Instruct.
#
# This generates the synthetic evaluation data used for accuracy benchmarks.
# Generated data is stored in data/ruler/data/llama-3/.
#
# Usage:
#   bash examples/build_ruler_data.sh
set -euo pipefail
cd "$(dirname "$0")/../data/ruler"

# Ensure NLTK punkt tokenizer is available
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

echo "Building RULER datasets for Llama-3.1-8B-Instruct..."
bash create_dataset.sh meta-llama/Meta-Llama-3.1-8B-Instruct llama-3
echo "Done. Data saved to data/ruler/data/llama-3/"
