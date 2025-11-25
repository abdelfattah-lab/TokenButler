

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


