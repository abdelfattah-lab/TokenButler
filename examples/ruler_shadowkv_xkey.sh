# --------- sparse budget 2048 ---------

# -------------- rank 48 ---------------
# ShadowKV
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv --sparse_budget 2048 --chunk_size 8 --rank 48 | tee -a logs/ruler/shadowkv_rank-48_sparse-2048.log

# ShadowKV xKey-1
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv_xkey --sparse_budget 2048 --chunk_size 8 --group_size 1 --rank_k 48 | tee -a logs/ruler/shadowkv_xkey-1_rank-48_sparse-2048.log

# ShadowKV xKey-2
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 2 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name "ruler/qa_2" \
    --method shadowkv_xkey --sparse_budget 2048 --chunk_size 8 --group_size 2 --rank_k 96 | tee -a logs/ruler/shadowkv_xkey-2_rank-96_sparse-2048.log

# ShadowKV xKey-4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv_xkey --sparse_budget 2048 --chunk_size 8 --group_size 4 --rank_k 192 | tee -a logs/ruler/shadowkv_xkey-4_rank-192_sparse-2048.log


# -------------- rank 64 ---------------
# ShadowKV
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv --sparse_budget 2048 --chunk_size 8 --rank 64 | tee -a logs/ruler/shadowkv_rank-64_sparse-2048.log

# ShadowKV xKey-1
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv_xkey --sparse_budget 2048 --chunk_size 8 --group_size 1 --rank_k 64 | tee -a logs/ruler/shadowkv_xkey-1_rank-64_sparse-2048.log

# ShadowKV xKey-2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv_xkey --sparse_budget 2048 --chunk_size 8 --group_size 2 --rank_k 128 | tee -a logs/ruler/shadowkv_xkey-2_rank-128_sparse-2048.log

# ShadowKV xKey-4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv_xkey --sparse_budget 2048 --chunk_size 8 --group_size 4 --rank_k 256 | tee -a logs/ruler/shadowkv_xkey-4_rank-256_sparse-2048.log

# -------------- rank 128 ---------------
# ShadowKV
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv --sparse_budget 2048 --chunk_size 8 --rank 128 | tee -a logs/ruler/shadowkv_rank-128_sparse-2048.log

# ShadowKV xKey-1
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv_xkey --sparse_budget 2048 --chunk_size 8 --group_size 1 --rank_k 128 | tee -a logs/ruler/shadowkv_xkey-1_rank-128_sparse-2048.log

# ShadowKV xKey-2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv_xkey --sparse_budget 2048 --chunk_size 8 --group_size 2 --rank_k 256 | tee -a logs/ruler/shadowkv_xkey-2_rank-256_sparse-2048.log

# ShadowKV xKey-4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv_xkey --sparse_budget 2048 --chunk_size 8 --group_size 4 --rank_k 512 | tee -a logs/ruler/shadowkv_xkey-4_rank-512_sparse-2048.log