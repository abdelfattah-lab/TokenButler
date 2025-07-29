
python main.py \
    --proj_name TokenButler_LayerPlacement \
    --model_path meta-llama/Llama-3.2-3B \
    --token_sparse_method fixed_40pc \
    --model_mode eval \
    --finetune_dataset redpajama \
    --train_subset_fac 5 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "L3_3B_PL0.csv" \
    --wname L3_3B_PL0 \
    --pred_source_layer 0 \
    --no_wandb \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --train_batch_size 4 --grad_accum_steps 16 \
    --model_load_path 
  
python main.py \
    --proj_name TokenButler_LayerPlacement \
    --model_path meta-llama/Llama-3.2-3B \
    --token_sparse_method fixed_40pc \
    --model_mode eval \
    --finetune_dataset redpajama \
    --train_subset_fac 5 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "L3_3B_PL4.csv" \
    --wname L3_3B_PL4 \
    --pred_source_layer 4 \
    --no_wandb \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --train_batch_size 4 --grad_accum_steps 16 \
    --model_load_path /home/ya255/projects/TokenButler/expt_model/TokenButler_LayerPlacement_42_finetune_None_None_500_llama_meta-llama_Llama-3.2-3B_L3_1B_PL4.csv_L3_1B_PL4_False_False_500_False_redpajama_1024_4_5_10_0.001_16_1024_16_False/4_1000_ExpPred_fixed_40pc_False_False_0_False_False_False_None_False_False_4_8_2_32_1024_4_False_False_True_28_0.07142857142857142_20250729-144438.pt
  
  
python main.py \
    --proj_name TokenButler_LayerPlacement \
    --model_path meta-llama/Llama-3.2-3B \
    --token_sparse_method fixed_40pc \
    --model_mode eval \
    --finetune_dataset redpajama \
    --train_subset_fac 5 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "L3_3B_PL8.csv" \
    --wname L3_3B_PL8 \
    --pred_source_layer 8 \
    --no_wandb \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --train_batch_size 4 --grad_accum_steps 16 \
    --model_load_path /home/ya255/projects/TokenButler/expt_model/TokenButler_LayerPlacement_42_finetune_None_None_500_llama_meta-llama_Llama-3.2-3B_L3_1B_PL8.csv_L3_1B_PL8_False_False_500_False_redpajama_1024_4_5_10_0.001_16_1024_16_False/4_1000_ExpPred_fixed_40pc_False_False_0_False_False_False_None_False_False_4_8_2_32_1024_8_False_False_True_28_0.12857142857142856_20250729-143846.pt
  
  
  
python main.py \
    --proj_name TokenButler_LayerPlacement \
    --model_path meta-llama/Llama-3.2-3B \
    --token_sparse_method fixed_40pc \
    --model_mode eval \
    --finetune_dataset redpajama \
    --train_subset_fac 5 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "L3_3B_PL16.csv" \
    --wname L3_3B_PL16 \
    --pred_source_layer 16 \
    --no_wandb \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --train_batch_size 4 --grad_accum_steps 16 \
    --model_load_path /home/ya255/projects/TokenButler/expt_model/TokenButler_LayerPlacement_42_finetune_None_None_500_llama_meta-llama_Llama-3.2-3B_L3_1B_PL16.csv_L3_1B_PL16_False_False_500_False_redpajama_1024_4_5_10_0.001_16_1024_16_False/4_1000_ExpPred_fixed_40pc_False_False_0_False_False_False_None_False_False_4_8_2_32_1024_16_False_False_True_28_0.2428571428571429_20250729-141913.pt
  
  
  
python main.py \
    --proj_name TokenButler_LayerPlacement \
    --model_path meta-llama/Llama-3.2-3B \
    --token_sparse_method fixed_40pc \
    --model_mode eval \
    --finetune_dataset redpajama \
    --train_subset_fac 5 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "L3_3B_PL24.csv" \
    --wname L3_3B_PL24 \
    --pred_source_layer 24 \
    --no_wandb \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --train_batch_size 4 --grad_accum_steps 16 \
    --model_load_path /home/ya255/projects/TokenButler/expt_model/TokenButler_LayerPlacement_42_finetune_None_None_500_llama_meta-llama_Llama-3.2-3B_L3_1B_PL24.csv_L3_1B_PL24_False_False_500_False_redpajama_1024_4_5_10_0.001_16_1024_16_False/4_1000_ExpPred_fixed_40pc_False_False_0_False_False_False_None_False_False_4_8_2_32_1024_24_False_False_True_28_0.35714285714285726_20250729-135240.pt
  