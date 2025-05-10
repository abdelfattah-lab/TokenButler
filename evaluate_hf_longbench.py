import argparse
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
from torch.cuda.amp import autocast


# Import your helpers from longbench_utils
from longbench_utils import scorer, MODEL2MAXLEN, DATASET2PROMPT, DATASET2MAXLEN

def build_chat(tokenizer, prompt, model_name):
    # Copy from KIVI
    if "longchat" in model_name.lower() or "vicuna" in model_name.lower():
        try:
            from fastchat.model import get_conversation_template
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        except ImportError:
            pass  # FastChat not installed
    elif "mistral-v0.2-instruct" in model_name.lower():
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name):
    preds = []
    for json_obj in tqdm.tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # Truncate to fit max_length
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        # Adjust if necessary
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt")
        context_length = input_ids.input_ids.shape[-1]
        embed_device = model.model.embed_tokens.weight.device
        with autocast():
            if dataset == "samsum":
                output = model.generate(
                    **input_ids.to(embed_device),
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0].to("cpu")
            else:
                output = model.generate(
                    **input_ids.to(embed_device),
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )[0].to("cpu")
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj.get("all_classes", None), "length": json_obj.get("length", None)})
    return preds

def run_long_bench_evaluation(model, tokenizer, args):
    """
    Runs the LongBench evaluation on specified datasets.
    """
    device = next(model.parameters()).device
    model.eval()
    model_name = args.model_path.lower()
    model_type = args.model_path.split("/")[-1].split('_')[0].lower()

    if model_type not in MODEL2MAXLEN:
        raise ValueError(f"Model type '{model_type}' not supported")

    max_length = MODEL2MAXLEN.get(model_type, 2048)
    print(f"Running LongBench evaluation on model: {model_name}")
    print(f"Max length: {max_length}")
    datasets = args.longbench_datasets
    dataset2prompt = DATASET2PROMPT
    dataset2maxlen = DATASET2MAXLEN

    results = {}
    for dataset in datasets:
        print(f"Evaluating dataset: {dataset}")
        start_time = time.time()
        data = load_dataset('THUDM/LongBench', dataset, split='test')

        prompt_format = dataset2prompt.get(dataset, "{text}")
        max_gen = dataset2maxlen.get(dataset, 256)

        preds = get_pred(
            model, tokenizer, data,
            max_length, max_gen,
            prompt_format, dataset,
            device, model_name
        )

        elapsed_time = time.time() - start_time
        print(f"Elapsed time for dataset {dataset}: {elapsed_time/60:.2f} minutes")

        predictions, answers, lengths = [], [], []
        all_classes = None
        for pred in preds:
            predictions.append(pred["pred"])
            answers.append(pred["answers"])
            if "length" in pred:
                lengths.append(pred["length"])
            all_classes = pred.get("all_classes", None)

        score = scorer(dataset, predictions, answers, all_classes)
        print(f"Dataset: {dataset} | Score: {score}")
        results[dataset] = score

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path or HuggingFace hub name of the model")
    parser.add_argument("--longbench_datasets", type=str, nargs="+", required=True,
                        help="List of LongBench datasets to evaluate on (e.g. hotpot_qa, narrative_qa)")
    parser.add_argument("--eval_subset", action="store_true",
                        help="Whether to evaluate on a 1% subset for speed")

    args = parser.parse_args()

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Run evaluation
    results = run_long_bench_evaluation(model, tokenizer, args)

    print("\n=== Final Results ===")
    for dataset, score in results.items():
        print(f"{dataset}: {score}")


if __name__ == "__main__":
    main()
