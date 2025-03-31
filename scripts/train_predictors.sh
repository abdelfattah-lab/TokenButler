### deepseek-ai/DeepSeek-R1-Distill-Llama-8B
python main.py \
    --proj_name TokenButler_March17 \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --token_sparse_method fixed_40pc \
    --model_mode finetune \
    --finetune_dataset redpajama \
    --train_subset_fac 1 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "L3_8B_R1_1K_BS2.csv" \
    --wname L3_8B_R1_1K_BS2 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --train_batch_size 2 \
    --do_downstream_eval \
    --task_list "winogrande,hellaswag,piqa,arc_easy" \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --softmax_causal_loss_ce --model_parallelism

### Llama-3.2-1B Training Script
python main.py \
    --proj_name TokenButler_March17 \
    --model_path meta-llama/Llama-3.2-1B \
    --architecture llama \
    --token_sparse_method fixed_40pc \
    --model_mode finetune \
    --finetune_dataset redpajama \
    --train_subset_fac 1 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "L3_1B_2k.csv" \
    --wname L3_1B_2k \
    --pred_lr 1e-3 \
    --dDash 16 \
    --intdim 512 \
    --do_downstream_eval \
    --task_list "winogrande,hellaswag,piqa,arc_easy" \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --softmax_causal_loss_ce


### Llama-3.2-3B Training Script
python main.py \
    --proj_name TokenButler_March17 \
    --model_path meta-llama/Llama-3.2-3B \
    --architecture llama \
    --token_sparse_method fixed_40pc \
    --model_mode finetune \
    --finetune_dataset redpajama \
    --train_subset_fac 1 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "L3_3B_2k.csv" \
    --wname L3_3B_2k \
    --pred_lr 1e-3 \
    --dDash 16 \
    --intdim 1024 \
    --do_downstream_eval \
    --task_list "winogrande,hellaswag,piqa,arc_easy" \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --softmax_causal_loss_ce


### Llama-2-7b-hf Training Script
python main.py \
    --proj_name TokenButler_March17 \
    --model_path meta-llama/Llama-2-7b-hf \
    --architecture llama \
    --token_sparse_method fixed_40pc \
    --model_mode finetune \
    --finetune_dataset redpajama \
    --train_subset_fac 1 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "L2_7B_2k.csv" \
    --wname L2_7B_2k \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --do_downstream_eval \
    --task_list "winogrande,hellaswag,piqa,arc_easy" \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --softmax_causal_loss_ce

### Llama-3.1-8B Training Script
python main.py \
    --proj_name TokenButler_March17 \
    --model_path meta-llama/Llama-3.1-8B \
    --token_sparse_method fixed_40pc \
    --model_mode finetune \
    --finetune_dataset redpajama \
    --train_subset_fac 1 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "L3_8B_1k_BS2.csv" \
    --wname L3_8B_1k_BS2 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --do_downstream_eval \
    --task_list "winogrande,hellaswag,piqa,arc_easy" \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024 \
    --train_batch_size 2 \
    --softmax_causal_loss_ce --model_parallelism
  


###  Do not bother retraining the models below.
    
### Mistral 7B v0.1 Training Script
python main.py \
    --proj_name TokenButler_March17 \
    --model_path mistralai/Mistral-7B-v0.1 \
    --architecture mistral \
    --token_sparse_method fixed_40pc \
    --model_mode finetune \
    --finetune_dataset redpajama \
    --train_subset_fac 1 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "M7B_1k.csv" \
    --wname M7B_1k \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --do_downstream_eval \
    --task_list "winogrande,hellaswag,piqa,arc_easy" \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024


### Phi-3.5-mini-instruct Training Script
python main.py \
    --proj_name TokenButler_March17 \
   --model_path  microsoft/Phi-3.5-mini-instruct        \
    --architecture phi3 \
    --token_sparse_method fixed_40pc \
    --model_mode finetune \
    --finetune_dataset redpajama \
    --train_subset_fac 1 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "P35mini_2k.csv" \
    --wname P35mini_2k \
    --pred_lr 1e-3 \
    --dDash 16 \
    --intdim 1024 \
    --do_downstream_eval \
    --task_list "winogrande,hellaswag,piqa,arc_easy" \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024


### Phi-3-mini-4k-instruct Training Script
python main.py \
    --proj_name TokenButler_March17 \
   --model_path  microsoft/Phi-3-mini-4k-instruct        \
    --architecture phi3 \
    --token_sparse_method fixed_40pc \
    --model_mode finetune \
    --finetune_dataset redpajama \
    --train_subset_fac 1 \
    --train_seqlen 1024 \
    --eval_llm_mode ExpPred \
    --result_file "P3mini_2k.csv" \
    --wname P3mini_2k \
    --pred_lr 1e-3 \
    --dDash 16 \
    --intdim 1024 \
    --do_downstream_eval \
    --task_list "winogrande,hellaswag,piqa,arc_easy" \
    --eval_subset 1000 \
    --eval_wk2_seqlen 1024

### potential buggy due to SFPA ###


# ### Qwen2.5-3B Training Script (Potentially buggy!)
# python main.py \
#     --proj_name TokenButler_March17 \
#    --model_path Qwen/Qwen2.5-3B  --architecture qwen \
#     --token_sparse_method fixed_40pc \
#     --model_mode finetune \
#     --finetune_dataset redpajama \
#     --train_subset_fac 1 \
#     --train_seqlen 1024 \
#     --eval_llm_mode ExpPred \
#     --result_file "Q25_3B_2k.csv" \
#     --wname Q25_3B_2k \
#     --pred_lr 1e-3 \
#     --dDash 32 \
#     --intdim 768 \
#     --do_downstream_eval \
#     --task_list "winogrande,hellaswag,piqa,arc_easy" \
#     --eval_subset 1000 \
#     --eval_wk2_seqlen 1024

# ### Qwen2.5-7B Training Script (Potentially buggy!)
# python main.py \
#     --proj_name TokenButler_March17 \
#    --model_path Qwen/Qwen2.5-7B  --architecture qwen \
#     --token_sparse_method fixed_40pc \
#     --model_mode finetune \
#     --finetune_dataset redpajama \
#     --train_subset_fac 1 \
#     --train_seqlen 1024 \
#     --eval_llm_mode ExpPred \
#     --result_file "Q25_3B_2k.csv" \
#     --wname Q25_3B_2k \
#     --pred_lr 1e-3 \
#     --dDash 32 \
#     --intdim 1280 \
#     --do_downstream_eval \
#     --task_list "winogrande,hellaswag,piqa,arc_easy" \
#     --eval_subset 1000 \
#     --eval_wk2_seqlen 1024