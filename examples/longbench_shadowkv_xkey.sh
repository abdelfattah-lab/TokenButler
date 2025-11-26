# --------- sparse budget 1024 ---------

# -------------- rank 64 ---------------
# ShadowKV
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method shadowkv --sparse_budget 1024 --chunk_size 8 --rank 64 | tee -a logs/longbench/shadowkv_rank-64_sparse-1024.log

# ShadowKV xKey-1
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method shadowkv_xkey --sparse_budget 1024 --chunk_size 8 --group_size 1 --rank_k 64 | tee -a logs/longbench/shadowkv_xkey-1_rank-64_sparse-1024.log

# ShadowKV xKey-2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method shadowkv_xkey --sparse_budget 1024 --chunk_size 8 --group_size 2 --rank_k 128 | tee -a logs/longbench/shadowkv_xkey-2_rank-128_sparse-1024.log

# ShadowKV xKey-4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method shadowkv_xkey --sparse_budget 1024 --chunk_size 8 --group_size 4 --rank_k 256 | tee -a logs/longbench/shadowkv_xkey-4_rank-256_sparse-1024.log

# -------------- rank 128 ---------------
# ShadowKV
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method shadowkv --sparse_budget 1024 --chunk_size 8 --rank 128 | tee -a logs/longbench/shadowkv_rank-128_sparse-1024.log

# ShadowKV xKey-1
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method shadowkv_xkey --sparse_budget 1024 --chunk_size 8 --group_size 1 --rank_k 128 | tee -a logs/longbench/shadowkv_xkey-1_rank-128_sparse-1024.log

# ShadowKV xKey-2
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method shadowkv_xkey --sparse_budget 1024 --chunk_size 8 --group_size 2 --rank_k 256 | tee -a logs/longbench/shadowkv_xkey-2_rank-256_sparse-1024.log

# ShadowKV xKey-4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method shadowkv_xkey --sparse_budget 1024 --chunk_size 8 --group_size 4 --rank_k 512 | tee -a logs/longbench/shadowkv_xkey-4_rank-512_sparse-1024.log