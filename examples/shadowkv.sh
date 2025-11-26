# -------------- rank 96, budget 2048, Llama-3.1-8B-Instruct ---------------
# ShadowKV - Ruler
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2 \
    --method shadowkv --sparse_budget 2048 --chunk_size 8 --rank 96 | tee -a logs/ruler_llama3/shadowkv_rank-96_sparse-2048.log

# ShadowKV - LongBench
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method shadowkv --sparse_budget 2048 --chunk_size 8 --rank 96 | tee -a logs/longbench_llama3/shadowkv_rank-96_sparse-2048.log_correct


# -------------- rank 96, budget 2048, Qwen2.5-7B-Instruct-1M ---------------
# ShadowKV - Ruler
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 \
    --dataset_name ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2 \
    --method shadowkv --sparse_budget 2048 --chunk_size 8 --rank 96 | tee -a logs/ruler_qwen/shadowkv_rank-96_sparse-2048.log

# ShadowKV - LongBench
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 test/eval_acc.py \
    --model_name Qwen/Qwen2.5-7B-Instruct-1M --datalen 65536 \
    --dataset_name long_bench/narrativeqa,long_bench/qasper,long_bench/multifieldqa_en,long_bench/hotpotqa,long_bench/2wikimqa,long_bench/musique,long_bench/gov_report,long_bench/qmsum,long_bench/multi_news,long_bench/trec,long_bench/triviaqa,long_bench/samsum,long_bench/passage_count,long_bench/passage_retrieval_en,long_bench/lcc,long_bench/repobench-p \
    --method shadowkv --sparse_budget 2048 --chunk_size 8 --rank 96 | tee -a logs/longbench_qwen/shadowkv_rank-96_sparse-2048.log

