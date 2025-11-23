
# python main.py \
#     --proj_name TokenButler_14Nov \
#     --model_path meta-llama/Llama-3.1-8B-Instruct \
#     --token_sparse_method fixed_40pc \
#     --model_mode finetune \
#     --finetune_dataset custom_mix_long \
#     --train_subset_fac 1 \
#     --train_seqlen 16384 \
#     --eval_llm_mode ExpPred \
#     --grad_accum_steps 8 \
#     --result_file "L3_8BiLong.csv" \
#     --wname L3_8BiLong \
#     --pred_lr 5e-4 \
#     --train_batch_size 1 \
#     --dDash 32 \
#     --intdim 1024 \
#     --eval_subset 1000 \
#     --eval_wk2_seqlen 16384 \
#     --train_batch_size 1 \
#     --predictor_init_path  /home/ya255/projects/TokenButler/checkpoints/TokenButler_14Nov_42_finetune_None_None_500_llama_meta-llama_Llama-3.1-8B-Instruct_L3_8Bi.csv_L3_8Bi_False_False_2000_False_custom_mix_1024_1_1_10_0.001_8_1024_16_False/4_False_False_True_32_0.3875000000000002.pt \
#     --softmax_causal_loss_ce --model_parallelism

    
python main.py \
    --proj_name TokenButler_14Nov \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --token_sparse_method fixed_40pc \
    --model_mode finetune \
    --finetune_dataset custom_mix_long \
    --train_subset_fac 1 \
    --train_seqlen 16384 \
    --eval_llm_mode ExpPred \
    --grad_accum_steps 8 \
    --result_file "L3_8BiLong_InitV2_PairCE.csv" \
    --wname L3_8BiLong_InitV2_PairCE \
    --pred_lr 1e-3 \
    --train_batch_size 1 \
    --dDash 32 \
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 16384 \
    --train_batch_size 1 \
    --pairwise_ce_loss --model_parallelism
    
python main.py \
    --proj_name TokenButler_14Nov \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --token_sparse_method fixed_40pc \
    --model_mode finetune \
    --finetune_dataset custom_mix_long \
    --train_subset_fac 1 \
    --train_seqlen 16384 \
    --eval_llm_mode ExpPred \
    --grad_accum_steps 8 \
    --result_file "L3_8BiLong_InitV2.csv" \
    --wname L3_8BiLong_InitV2 \
    --pred_lr 1e-3 \
    --train_batch_size 1 \
    --dDash 32 \
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 16384 \
    --train_batch_size 1 \
    --softmax_causal_loss_ce --model_parallelism


# /mnt/home/ya255/projects/TokenButler/expt_model/TokenButler_14Nov_42_finetune_None_None__mnt_home_ya255_projects_TokenButler_checkpoints_TokenButler_14Nov_42_finetune_None_None_500_llama_meta-llama_Llama-3.1-8B-Instruct_L3_8Bi.csv_L3_8Bi_F_cb1cc44e/4_1000_ExpPred_fixed_40pc_False_False_0_False_False_True_False_False_None_False_False_4_8_2_32_1024_False_False_True_32_0.3875000000000002_20251119-143523.pt