#!/bin/bash
# Download TokenButler predictor weights from HuggingFace.
#
# Usage:
#   bash scripts/download_weights.sh
#
# Requires: huggingface-cli (pip install huggingface_hub)
set -euo pipefail
cd "$(dirname "$0")/.."

WEIGHTS_FILE="L3_8Bi_d16_i512_pf4.pt"
REPO_ID="AhmedAE/xKV-Llama-3.1-8B-Instruct"

if [ -f "$WEIGHTS_FILE" ]; then
    echo "Weights already exist at $WEIGHTS_FILE"
    exit 0
fi

echo "Downloading $WEIGHTS_FILE from $REPO_ID..."
huggingface-cli download "$REPO_ID" "$WEIGHTS_FILE" --local-dir .
echo "Done. Saved to $WEIGHTS_FILE"
