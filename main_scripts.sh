

python main.py \
    --proj_name TokenButler_24Nov \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --token_sparse_method fixed_40pc \
    --model_mode finetune \
    --finetune_dataset custom_mix_long \
    --train_subset_fac 1 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --grad_accum_steps 16 \
    --result_file "L3_8BiLong_p4x.csv" \
    --wname L3_8BiLong_p4x \
    --pred_lr 1e-3 \
    --train_batch_size 1 \
    --dDash 16 \
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --train_batch_size 1 \
    --softmax_causal_loss_ce \
    --tokenbutler_project \
    --producer_frequency 4


python main.py \
    --proj_name TokenButler_24Nov \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --token_sparse_method fixed_40pc \
    --model_mode finetune \
    --finetune_dataset custom_mix_long \
    --train_subset_fac 1 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --grad_accum_steps 16 \
    --result_file "L3_8BiLong_p.csv" \
    --wname L3_8BiLong_p \
    --pred_lr 1e-3 \
    --train_batch_size 1 \
    --dDash 16 \
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --train_batch_size 1 \
    --softmax_causal_loss_ce \
    --tokenbutler_project 




python main.py \
    --proj_name TokenButler_24Nov \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --token_sparse_method fixed_40pc \
    --model_mode finetune \
    --finetune_dataset custom_mix_long \
    --train_subset_fac 1 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --grad_accum_steps 16 \
    --result_file "L3_8BiLong_s4x.csv" \
    --wname L3_8BiLong_s4x \
    --pred_lr 1e-3 \
    --train_batch_size 1 \
    --dDash 16 \
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --train_batch_size 1 \
    --softmax_causal_loss_ce \
    --tokenbutler_slice \
    --producer_frequency 4


#### Evaluation



CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nnodes=1 --nproc_per_node 2 \
  longeval/eval_tokenbutler.py \
  --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --architecture llama \
  --datalen 16384 \
  --dataset_name "long_bench/qasper" \
  --dDash 16 \
  --intdim 1024 \
  --result_dir results_tokbutler/long_run_partial_p4x \
  --eval_llm_mode ExpPred \
  --token_sparse_method fixed_2048tok \
  --min_sparse_index 128 \
  --sliding_window 512 \
  --tokenbutler_project \
  --producer_frequency 4 \
  --predictor_ckpt /home/ya255/projects/TokenButler/checkpoints/TokenButler_24Nov_42_finetune_None_None_None_500_llama_meta-llama_Llama-3.1-8B-Instruct_L3_8BiLong_p4x.csv_L3_8BiLong_p4x_False_False_2000_4_False_custom_mix_long_1024_1_1_10_0.001_16_1024_16_False_4/e_tokenbutler_project_0.3875000000000002.pt



#   --dataset_name "ruler/fwe,ruler/qa_1,ruler/qa_2,ruler/vt" \
                # "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,\
                # ruler/niah_multikey_1,ruler/niah_multikey_2,\
                # ruler/niah_multiquery,ruler/niah_multivalue,\

python test_generation.py \
    --proj_name TrainTokenButler \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --architecture llama \
    --token_sparse_method fixed_512tok \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 800 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "TEST_GENERATION.csv" \
    --wname TEST_GENERATION \
    --no_wandb \
    --pred_lr 1e-3 \
    --dDash 16 \
    --sliding_window 64 \
    --tokenbutler_project \
    --producer_frequency 4 \
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --predictor_ckpt /home/ya255/projects/TokenButler/checkpoints/TokenButler_24Nov_42_finetune_None_None_None_500_llama_meta-llama_Llama-3.1-8B-Instruct_L3_8BiLong_p4x.csv_L3_8BiLong_p4x_False_False_2000_4_False_custom_mix_long_1024_1_1_10_0.001_16_1024_16_False_4/e_tokenbutler_project_0.3875000000000002.pt


