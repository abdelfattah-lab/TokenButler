#!/bin/bash

# Batch Size Sweep for 60k Sequence Length
# Fixed Parameters:
#   - Prompt Length: 60k (61440 tokens)
#   - Sparse Budget: 1024
#   - Gen Length: 100
# Variable Parameters:
#   - Batch Sizes: 8, 16, 32, 48
#   - Methods: baseline, shadowkv, xkey, xkv
# Rank Configuration:
#   - ShadowKV: rank_k=96
#   - xKey/xKV: rank_k=384, rank_v=576, group_size=4

PROMPT_LEN=122880  # 120k tokens
SPARSE_BUDGET=1024
GEN_LEN=100
BATCH_SIZES=(16)
NUM_ROUNDS=3  # Number of rounds for averaging

# ShadowKV parameters
SHADOWKV_RANK_K=96

# xKey/xKV parameters  
XKV_RANK_K=384
XKV_RANK_V=512
GROUP_SIZE=4

# Create logs directory if it doesn't exist
mkdir -p logs/sweep_120k
mkdir -p logs/results

echo "========================================"
echo "Starting Batch Size Sweep for 120k"
echo "========================================"
echo ""

# Loop through batch sizes
for BSZ in "${BATCH_SIZES[@]}"; do
    echo "========================================"
    echo "Testing Batch Size: $BSZ"
    echo "========================================"
    
    # # Test Baseline (Full Attention)
    # echo ""
    # echo "--- Running Baseline (Full Attention) ---"
    # python test/e2e.py \
    #     --prompt_len $PROMPT_LEN \
    #     --bsz $BSZ \
    #     --budget $SPARSE_BUDGET \
    #     --gen_len $GEN_LEN \
    #     --num_rounds $NUM_ROUNDS \
    #     --baseline \
    #     2>&1 | tee -a logs/sweep_120k/baseline_bs${BSZ}.log
    
    # sleep 2  # Brief pause between runs
    
    # Test ShadowKV
    echo ""
    echo "--- Running ShadowKV (rank_k=$SHADOWKV_RANK_K) ---"
    python test/e2e.py \
        --prompt_len $PROMPT_LEN \
        --bsz $BSZ \
        --budget $SPARSE_BUDGET \
        --gen_len $GEN_LEN \
        --num_rounds $NUM_ROUNDS \
        --shadowkv \
        --rank_k $SHADOWKV_RANK_K \
        2>&1 | tee -a logs/sweep_120k/shadowkv_k${SHADOWKV_RANK_K}_bs${BSZ}.log
    
    sleep 2
    
    # Test xKey
    echo ""
    echo "--- Running xKey (rank_k=$XKV_RANK_K, rank_v=$XKV_RANK_V, group_size=$GROUP_SIZE) ---"
    python test/e2e.py \
        --prompt_len $PROMPT_LEN \
        --bsz $BSZ \
        --budget $SPARSE_BUDGET \
        --gen_len $GEN_LEN \
        --num_rounds $NUM_ROUNDS \
        --xkey \
        --rank_k $XKV_RANK_K \
        --rank_v $XKV_RANK_V \
        --group_size $GROUP_SIZE \
        2>&1 | tee -a logs/sweep_120k/xkey_k${XKV_RANK_K}_v${XKV_RANK_V}_g${GROUP_SIZE}_bs${BSZ}.log
    
    # sleep 2
    
    # Test xKV
    echo ""
    echo "--- Running xKV (rank_k=$XKV_RANK_K, rank_v=$XKV_RANK_V, group_size=$GROUP_SIZE) ---"
    python test/e2e.py \
        --prompt_len $PROMPT_LEN \
        --bsz $BSZ \
        --budget $SPARSE_BUDGET \
        --gen_len $GEN_LEN \
        --num_rounds $NUM_ROUNDS \
        --xkv \
        --rank_k $XKV_RANK_K \
        --rank_v $XKV_RANK_V \
        --group_size $GROUP_SIZE \
        2>&1 | tee -a logs/sweep_120k/xkv_k${XKV_RANK_K}_v${XKV_RANK_V}_g${GROUP_SIZE}_bs${BSZ}.log
    
    # echo ""
    # echo "Completed Batch Size: $BSZ"
    # echo "========================================"
    # echo ""
    
    # sleep 5  # Longer pause between batch size changes
done

echo ""
echo "========================================"
echo "Sweep Complete!"
echo "========================================"
echo ""
echo "Results saved in logs/sweep_120k/ and logs/results/"
echo ""
echo "Summary of configurations tested:"
echo "  Prompt Length: $PROMPT_LEN"
echo "  Batch Sizes: ${BATCH_SIZES[*]}"
echo "  Number of Rounds: $NUM_ROUNDS"
echo "  ShadowKV: rank_k=$SHADOWKV_RANK_K"
echo "  xKey/xKV: rank_k=$XKV_RANK_K, rank_v=$XKV_RANK_V, group_size=$GROUP_SIZE"

