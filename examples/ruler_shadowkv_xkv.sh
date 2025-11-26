# --------- rank 96, sparse budget 2048, Llama-3.1-8B-Instruct ---------
# ShadowKV xKV-2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv_xkv --sparse_budget 2048 --group_size 2 --rank_k 192 --rank_v 288 --chunk_size 8 | tee -a logs/ruler_llama3/shadowkv_xkv-2_k192_v288_sparse-2048.log

# ShadowKV xKV-4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv_xkv --sparse_budget 2048 --group_size 4 --rank_k 384 --rank_v 576 --chunk_size 8 | tee -a logs/ruler_llama3/shadowkv_xkv-4_k384_v576_sparse-2048.log

# --------- rank 96, sparse budget 1024, Llama-3.1-8B-Instruct ---------
# ShadowKV xKV-2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv_xkv --sparse_budget 1024 --group_size 2 --rank_k 192 --rank_v 288 --chunk_size 8 | tee -a logs/ruler_llama3/shadowkv_xkv-2_k192_v288_sparse-1024.log

# ShadowKV xKV-4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv_xkv --sparse_budget 1024 --group_size 4 --rank_k 384 --rank_v 576 --chunk_size 8 | tee -a logs/ruler_llama3/shadowkv_xkv-4_k384_v576_sparse-1024.log


# --------- rank 96, sparse budget 2048, Qwen2.5-7B-Instruct-1M ---------
# ShadowKV xKV-2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv_xkv --sparse_budget 2048 --group_size 2 --rank_k 192 --rank_v 288 --chunk_size 8 | tee -a logs/ruler_qwen/shadowkv_xkv-2_k192_v288_sparse-2048.log

# ShadowKV xKV-4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv_xkv --sparse_budget 2048 --group_size 4 --rank_k 384 --rank_v 576 --chunk_size 8 | tee -a logs/ruler_qwen/shadowkv_xkv-4_k384_v576_sparse-2048.log

# --------- rank 96, sparse budget 1024, Qwen2.5-7B-Instruct-1M ---------
# ShadowKV xKV-2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv_xkv --sparse_budget 1024 --group_size 2 --rank_k 192 --rank_v 288 --chunk_size 8 | tee -a logs/ruler_qwen/shadowkv_xkv-2_k192_v288_sparse-1024.log

# ShadowKV xKV-4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --method shadowkv_xkv --sparse_budget 1024 --group_size 4 --rank_k 384 --rank_v 576 --chunk_size 8 | tee -a logs/ruler_qwen/shadowkv_xkv-4_k384_v576_sparse-1024.log
