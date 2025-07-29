# Dense Basline

python main.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --token_sparse_method fixed_10pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode dense \
    --result_file C4_L3_8B_R1_1K_Dense.csv \
    --wname C4_L3_8B_R1_1K_Dense \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --model_load_path "/home/ya255/projects/trained_models/Butler_DeepSeek-R1-Distill-Llama-8B.pt" \
    --do_downstream_eval \
    --task_list "leaderboard_mmlu_pro,leaderboard_bbh_causal_judgement"

# Predictor Runs

python main.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --token_sparse_method fixed_10pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode ExpPred \
    --result_file C4_L3_8B_R1_1K_Pred.csv \
    --wname C4_L3_8B_R1_1K_Pred \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --model_load_path "/home/ya255/projects/trained_models/Butler_DeepSeek-R1-Distill-Llama-8B.pt" \
    --do_downstream_eval \
    --task_list "leaderboard_mmlu_pro,leaderboard_bbh_causal_judgement"


python main.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --token_sparse_method fixed_30pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode ExpPred \
    --result_file C4_L3_8B_R1_1K_Pred.csv \
    --wname C4_L3_8B_R1_1K_Pred \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --model_load_path "/home/ya255/projects/trained_models/Butler_DeepSeek-R1-Distill-Llama-8B.pt" \
    --do_downstream_eval \
    --task_list "leaderboard_mmlu_pro,leaderboard_bbh_causal_judgement"

python main.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --token_sparse_method fixed_50pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode ExpPred \
    --result_file C4_L3_8B_R1_1K_Pred.csv \
    --wname C4_L3_8B_R1_1K_Pred \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --model_load_path "/home/ya255/projects/trained_models/Butler_DeepSeek-R1-Distill-Llama-8B.pt" \
    --do_downstream_eval \
    --task_list "leaderboard_mmlu_pro,leaderboard_bbh_causal_judgement"



python main.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --token_sparse_method fixed_70pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode ExpPred \
    --result_file C4_L3_8B_R1_1K_Pred.csv \
    --wname C4_L3_8B_R1_1K_Pred \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --model_load_path "/home/ya255/projects/trained_models/Butler_DeepSeek-R1-Distill-Llama-8B.pt" \
    --do_downstream_eval \
    --task_list "leaderboard_mmlu_pro,leaderboard_bbh_causal_judgement"


# Oracle Runs


python main.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --token_sparse_method fixed_10pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode oracle \
    --result_file C4_L3_8B_R1_1K_Oracle.csv \
    --wname C4_L3_8B_R1_1K_Oracle \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --model_load_path "/home/ya255/projects/trained_models/Butler_DeepSeek-R1-Distill-Llama-8B.pt" \
    --do_downstream_eval \
    --task_list "leaderboard_mmlu_pro,leaderboard_bbh_causal_judgement"


python main.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --token_sparse_method fixed_30pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode oracle \
    --result_file C4_L3_8B_R1_1K_Oracle.csv \
    --wname C4_L3_8B_R1_1K_Oracle \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --model_load_path "/home/ya255/projects/trained_models/Butler_DeepSeek-R1-Distill-Llama-8B.pt" \
    --do_downstream_eval \
    --task_list "leaderboard_mmlu_pro,leaderboard_bbh_causal_judgement"

python main.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --token_sparse_method fixed_50pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode oracle \
    --result_file C4_L3_8B_R1_1K_Oracle.csv \
    --wname C4_L3_8B_R1_1K_Oracle \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --model_load_path "/home/ya255/projects/trained_models/Butler_DeepSeek-R1-Distill-Llama-8B.pt" \
    --do_downstream_eval \
    --task_list "leaderboard_mmlu_pro,leaderboard_bbh_causal_judgement"


python main.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --token_sparse_method fixed_70pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode oracle \
    --result_file C4_L3_8B_R1_1K_Oracle.csv \
    --wname C4_L3_8B_R1_1K_Oracle \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --model_load_path "/home/ya255/projects/trained_models/Butler_DeepSeek-R1-Distill-Llama-8B.pt" \
    --do_downstream_eval \
    --task_list "leaderboard_mmlu_pro,leaderboard_bbh_causal_judgement"


#### BASELINES



python main.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --token_sparse_method fixed_50pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode quest \
    --result_file C4_L3_8B_R1_1K_quest.csv \
    --wname C4_L3_8B_R1_1K_quest \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --model_load_path "/home/ya255/projects/trained_models/Butler_DeepSeek-R1-Distill-Llama-8B.pt" \
    --do_downstream_eval \
    --task_list "leaderboard_mmlu_pro,leaderboard_bbh_causal_judgement"


python main.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --token_sparse_method fixed_50pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode snapkv \
    --result_file C4_L3_8B_R1_1K_snapkv.csv \
    --wname C4_L3_8B_R1_1K_snapkv \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --model_load_path "/home/ya255/projects/trained_models/Butler_DeepSeek-R1-Distill-Llama-8B.pt" \
    --do_downstream_eval \
    --task_list "leaderboard_mmlu_pro,leaderboard_bbh_causal_judgement"

python main.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --token_sparse_method fixed_50pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode magicpig \
    --result_file C4_L3_8B_R1_1K_magicpig.csv \
    --wname C4_L3_8B_R1_1K_magicpig \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --model_load_path "/home/ya255/projects/trained_models/Butler_DeepSeek-R1-Distill-Llama-8B.pt" \
    --do_downstream_eval \
    --no_wikitext_eval \
    --task_list "leaderboard_mmlu_pro,leaderboard_bbh_causal_judgement"



python main.py \
    --proj_name TokenButler_Reasoning \
    --architecture llama \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --token_sparse_method fixed_50pc \
    --model_mode eval \
    --finetune_dataset c4_realnewslike \
    --train_subset_fac 4 \
    --eval_llm_mode quest \
    --result_file C4_L3_8B_R1_1K_quest.csv \
    --wname C4_L3_8B_R1_1K_quest \
    --no_wandb \
    --sliding_window 4 \
    --eval_wk2_seqlen 1024 \
    --num_tok_per_page 4 \
    --pred_lr 1e-3 \
    --dDash 32 \
    --intdim 1024 \
    --model_load_path "/home/ya255/projects/trained_models/Butler_DeepSeek-R1-Distill-Llama-8B.pt" \
    --do_downstream_eval \
    --task_list "leaderboard_mmlu_pro,leaderboard_bbh_causal_judgement"

