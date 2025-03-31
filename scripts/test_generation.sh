python test_generation.py \
    --proj_name TrainTokenButler \
    --model_path meta-llama/Llama-3.2-3B \
    --architecture llama \
    --token_sparse_method fixed_40pc \
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
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --model_load_path "/home/ya255/projects/TokenButler/expt_model/TokenButler_March17_42_finetune_None_None_500_llama_meta-llama_Llama-3.2-3B_L3_3B_2k.csv_L3_3B_2k_False_False_2000_False_redpajama_1024_1_1_100_0.001_1024_16_False/4_1000_ExpPred_fixed_40pc_True_False_0_False_False_True_None_False_False_4_8_2_16_1024_False_False_True_28_0.38571428571428584_20250318-193424.pt" 

python test_generation.py \
    --proj_name TrainTokenButler \
    --model_path meta-llama/Llama-3.2-1B \
    --architecture llama \
    --token_sparse_method fixed_40pc \
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
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --model_load_path "/home/ya255/projects/TokenButler/expt_model/TokenButler_March17_42_finetune_None_None_500_llama_meta-llama_Llama-3.2-1B_L3_1B_2k.csv_L3_1B_2k_False_False_2000_False_redpajama_1024_1_1_100_0.001_1024_16_False/4_1000_ExpPred_fixed_40pc_True_False_0_False_False_True_None_False_False_4_8_2_16_512_False_False_True_16_0.37500000000000006_20250318-091102.pt" 



python test_generation.py \
    --proj_name TrainTokenButler \
    --model_path meta-llama/Llama-3.1-8B \
    --architecture llama \
    --token_sparse_method fixed_40pc \
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
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --model_load_path "/home/ya255/projects/TokenButler/expt_model/TokenButler_March17_42_finetune_None_None_500_llama_meta-llama_Llama-3.1-8B_L3_8B_1k_BS2.csv_L3_8B_1k_BS2_True_False_2000_False_redpajama_1024_2_1_100_0.001_1024_16_False/4_1000_ExpPred_fixed_40pc_True_False_0_False_False_True_None_False_False_4_8_2_32_1024_False_False_True_32_0.3875000000000002__best.pt" 




python test_generation.py \
    --proj_name TrainTokenButler \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --architecture llama \
    --token_sparse_method fixed_40pc \
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
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --model_load_path "/home/ya255/projects/TokenButler/expt_model/TokenButler_March17_42_finetune_None_None_500_llama_deepseek-ai_DeepSeek-R1-Distill-Llama-8B_L3_8B_R1_1K_BS2.csv_L3_8B_R1_1K_BS2_True_False_2000_False_redpajama_1024_2_1_100_0.001_1024_16_False/4_1000_ExpPred_fixed_40pc_True_False_0_False_False_True_None_False_False_4_8_2_32_1024_False_False_True_32_0.3875000000000002__best.pt" \




python test_generation.py \
    --proj_name TrainTokenButler \
    --model_path meta-llama/Llama-2-7b-hf \
    --architecture llama \
    --token_sparse_method fixed_40pc \
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
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --model_load_path "/home/ya255/projects/TokenButler/expt_model/TokenButler_March17_42_finetune_None_None_500_llama_meta-llama_Llama-2-7b-hf_L2_7B_2k.csv_L2_7B_2k_False_False_2000_False_redpajama_1024_1_1_100_0.001_1024_16_False/4_1000_ExpPred_fixed_40pc_True_False_0_False_False_True_None_False_False_4_8_2_32_1024_False_False_True_32_0.3875000000000002__best.pt" 



