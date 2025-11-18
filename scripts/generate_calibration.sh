
python main.py \
    --proj_name TokenButler_EvalDebugs \
    --no_wandb \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --token_sparse_method fixed_40pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "C_L3_8B_R1.csv" \
    --wname C_L3_8B_R1 \
    --pred_lr 1e-3 \
    --sliding_window 4 \
    --dDash 32 \
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 --num_tok_per_page 4 \
    --model_load_path /mnt/home/ya255/projects/TokenButler/checkpoints/TokenButler_14Nov_42_finetune_None_None_500_llama_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_L3_8B_R1.csv_L3_8B_R1_False_False_2000_False_custom_mix_1024_1_1_10_0.001_8_1024_16_False/4_False_False_True_32_0.3875000000000002.pt

    
python main.py \
    --proj_name TokenButler_EvalDebugs \
    --no_wandb \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --token_sparse_method fixed_40pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "C_L3_8B.csv" \
    --wname C_L3_8B \
    --pred_lr 1e-3 \
    --sliding_window 4 \
    --dDash 32 \
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 --num_tok_per_page 4 \
    --model_load_path /mnt/home/ya255/projects/TokenButler/checkpoints/TokenButler_14Nov_42_finetune_None_None_500_llama_meta-llama_Llama-3.1-8B-Instruct_L3_8Bi.csv_L3_8Bi_False_False_2000_False_custom_mix_1024_1_1_10_0.001_8_1024_16_False/4_False_False_True_32_0.3875000000000002.pt
    