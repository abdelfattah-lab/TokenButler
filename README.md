# TokenButler: Token Importance Is Predictable

This project is still WIP and under improvement. 

For reproducing prefill-style TokenButler paper results: Please checkout commit `d94fc426990089c351d062eb65005da2a8d115a2` for reproducing results from paper; note that the commit-id will work only for prefill tests due to decoding bug.

For better models that function with decode as well -- current main commit is best! The drive link for the latest Llama-class TokenButler models: [TokenButler_Models_31Mar.zip](https://drive.google.com/file/d/1w2wq23dy7vNA5Dn7an-5kHeNTYtc3HKB/view?usp=drive_link). Further, the latest models are also available on huggingface!

<div align="center">
  <img src="https://github.com/abdelfattah-lab/TokenButler/blob/main/figs/tokenbutlerlogo.png?raw=true" width="50%" alt="TokenButler" />
</div>

<br>
<div align="center" style="line-height: 1;">
  <!-- Paper Badge -->
  <a href="https://arxiv.org/abs/2503.07518" target="_blank" style="margin: 2px;">
    <img alt="Paper" 
         src="https://img.shields.io/badge/Paper-View-orange?logo=readthedocs&logoColor=white" 
         style="display: inline-block; vertical-align: middle;"/>
  </a>
  <!-- GitHub Badge -->
  <a href="https://github.com/abdelfattah-lab/TokenButler" target="_blank" style="margin: 2px;">
    <img alt="GitHub" 
         src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" 
         style="display: inline-block; vertical-align: middle;"/>
  </a>
  <!-- Huggingface Badge -->
  <a href="https://huggingface.co/collections/akhauriyash/tokenbutler-67cf181b5762d0d60e5f312b" target="_blank" style="margin: 2px;">
    <img alt="Huggingface" 
         src="https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000" 
         style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<br>


# Predictor Architecture

![Predictor Architecture](https://github.com/abdelfattah-lab/TokenButler/blob/main/figs/mainfig.png)

This repository contains code to train and evaluate 'token importance' predictors.

## Huggingface

We support the following models directly through huggingface-transformers:

- **DeepSeek-R1-Distill-Llama-8B-Butler**
- **Llama-3.1-8B-Butler**
- **Llama-2-7b-hf-Butler**
- **Llama-3.2-3B-Butler**
- **Llama-3.2-1B-Butler**

The collection of models can be found [here](https://huggingface.co/collections/akhauriyash/tokenbutler-67cf181b5762d0d60e5f312b)

Simply run `test_hf.py` or:

```
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

question = "If millionaires have butlers, why don't million dollar language models have a butler too? I think its because "

model_name = "akhauriyash/DeepSeek-R1-Distill-Llama-8B-Butler"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
response = generator(question, max_new_tokens=200, do_sample=True, top_p=0.95, temperature=0.7)

print(response[0]['generated_text'][len(question):])
```

Note that the 'default' configured sparsity is 50%. Further, there is a 'sliding window' of 128 and 8 'anchor tokens'. To 'change' the sparsity, you can use the following function after loading the model. Please note that the 'fixed' is the only supported strategy at the moment, which 'fixes' the sparsity of each layer (except the first) at the 'pc' (percentage) mentioned. This can also be found at `test_hf.py`. Sliding window and anchor tokens can be changed in a similar manner.

```
def set_sparsity(model, sparsity):
    for module in model.modules():
        if module.__class__.__name__.__contains__("AttentionExperimental"):
            module.token_sparse_method = sparsity
            module.set_token_sparsity()
    return model

model = set_sparsity(model, "fixed_60pc")
```

All of our results, traces from experiments are located in `ablation_results/`

Note: Our predictor design has improved since the arXiv paper release (We added a layer-norm to stabilize training). Further, to focus on the main predictor design and training-eval scripts, we have removed the ablation scripts. To reproduce the original results and predictor models, please checkout commit `0412fc24a3b770e4d82e6d7064a8172f24c5fcd3` and download the old models from [Drive Link](https://drive.google.com/drive/folders/1psNZ1SU0LaZJ-x5MQGH59CzYSmeT4yRf?usp=sharing).

For the latest, new models, try the huggingface integration. [Wandb-Logs](https://wandb.ai/akhauriyash/TrainTokenButler) for trained models.

## Installation

```
conda create --name TokenButler python=3.10
conda activate TokenButler
python -m pip install -r requirements.txt
```

## Evaluation

Please download our trained (old) TokenButler predictor models from this [Drive Link](https://drive.google.com/drive/folders/1psNZ1SU0LaZJ-x5MQGH59CzYSmeT4yRf?usp=sharing)

To evaluate, example scripts are provided in `scripts/eval_scan.sh`, checkout commit `0412fc24a3b770e4d82e6d7064a8172f24c5fcd3`. Decode-generation may not work at this commit.

### Example script (Please update checkpoint path after downloading models):
```
bash eval_scan.sh L3_3B_2k_1PC.csv L3_3B_2k_1PC ExpPred meta-llama/Llama-3.2-3B 1024 16 "<PATH TO CHECKPOINT>"
```

# Custom Synthetic Task

![Custom Synthetic Task](https://github.com/abdelfattah-lab/TokenButler/blob/main/figs/datasetfig.png)

### Modes supported: 
- **TokenButler:** `ExpPred`
- **Oracle:** `oracle`
- **H2O:** `h2o_true` (Generation not supported, prefill is 'decode simulated')
- **SnapKV:** `snapkv` (Generation not supported, prefill is 'decode simulated')
- **Quest:** `quest` (Generation not supported)


**Important note on our evaluation strategy:** To properly test token-eviction/selection methods in a longer decode setting, we _simulate_ token eviction based strategies (SnapKV and H2O) in a purely decode setting. For example, with a 50% token budget, we simulate the entire input sequence as if it were fully decoded, allocating tokens **proportionally to the sequence length** at each decode step. This approach helps profile prefill-eviction-based methods by accurately emulating their token eviction policies across the full input sequence. Unfortunately, this also makes accuracy evaluation slower.

To change downstream evaluation models, modify `task_list`, to evaluate on a smaller subset, modify `eval_subset`. To modify token sparsities being evaluated, modify `{10..60..10}` as desired.


# Training
Our training scripts are located in `scripts/train_predictors.sh`

We provide scripts for the following models:

- **deepseek-ai/DeepSeek-R1-Distill-Llama-8B**
- **meta-llama/Llama-3.2-1B**
- **meta-llama/Llama-3.2-3B**
- **meta-llama/Llama-2-7b-hf**
- **meta-llama/Llama-3.1-8B**
- **mistralai/Mistral-7B-v0.1**
- **Qwen/Qwen2.5-3B**
- **Qwen/Qwen2.5-7B**
- **microsoft/Phi-3.5-mini-instruct**
- **microsoft/Phi-3-mini-4k-instruct**

Training requires 1 A6000 GPU for these variants. Longer-context training is possible using --model_parallelism

# Reasoning Model TokenButler Results

Model: DeepSeek-R1-Distill-Llama-8B-Butler

|Method     |Sparsity (%)      |Perplexity|BBH Causal Judgement|MMLU-Pro           |
|-----------|------------------|----------|--------------------|-------------------|
|Dense      |0    |15.87     |0.55  |0.274 |
|TokenButler|12.2 |15.90     |0.56  |0.275 |
|TokenButler|31.0 |15.99     |0.55  |0.273 |
|TokenButler|49.8 |16.22     |0.56  |0.273 |
|TokenButler|68.2 |16.99     |0.55  |0.263 |
|Oracle     |12.2 |15.85     |0.56  |0.273 |
|Oracle     |31.0 |15.76     |0.55  |0.273 |
|Oracle     |49.8 |15.66     |0.54  |0.271 |
|Oracle     |68.3 |15.71     |0.51  |0.271 |

## Citation

```
@misc{akhauri2025tokenbutlertokenimportancepredictable,
      title={TokenButler: Token Importance is Predictable}, 
      author={Yash Akhauri and Ahmed F AbouElhamayed and Yifei Gao and Chi-Chih Chang and Nilesh Jain and Mohamed S. Abdelfattah},
      year={2025},
      eprint={2503.07518},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.07518}, 
}
```
