################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

from datasets import load_dataset
from termcolor import colored
import random
import numpy as np

# RULER & LongBench
from .metrics import (
    needle_score,
    string_match_part,
    multi_number, 
    multi_words,
    rouge_score,
    retrieval_score,
    code_sim_score,
    qa_f1_score,
    count_score,
    classification_score,
)

# NIAH
from data.utils import generate_random_number, read_context_files, create_contexts, NIAH_TEMPLATE, RANDOM_NEEDLE_CITIES, LONG_BENCH_TEMPLATE


METRICS_FN = {
    'niah': needle_score,
    'multi': multi_number,
    'vt': multi_words,
    'cwe': multi_words,
    'fwe': multi_words,
    'qa': string_match_part,
    
    # Single-Document QA
    "long_bench/narrativeqa": qa_f1_score,
    "long_bench/qasper": qa_f1_score,
    "long_bench/multifieldqa_en": qa_f1_score,
    # Multi-Document QA
    "long_bench/hotpotqa": qa_f1_score,
    "long_bench/2wikimqa": qa_f1_score,
    "long_bench/musique": qa_f1_score,
    # Summarization
    "long_bench/gov_report": rouge_score,
    "long_bench/qmsum": rouge_score,
    "long_bench/multi_news": rouge_score,
    # Few-shot Learning
    "long_bench/trec": classification_score,
    "long_bench/triviaqa": qa_f1_score,
    "long_bench/samsum": rouge_score,
    # Synthetic Task
    "long_bench/passage_count": count_score,
    "long_bench/passage_retrieval_en": retrieval_score,
    # Code Completion
    "long_bench/lcc": code_sim_score,
    "long_bench/repobench-p": code_sim_score,
}

GEN_LEN = {
    'niah': 64,
    'vt': 30,
    'cwe': 120,
    'fwe': 50,
    'qa': 32,
    
    "long_bench/narrativeqa": 128,
    "long_bench/qasper": 128,
    "long_bench/multifieldqa_en": 64,
    "long_bench/multifieldqa_zh": 64,
    "long_bench/hotpotqa": 32,
    "long_bench/2wikimqa": 32,
    "long_bench/musique": 32,
    "long_bench/dureader": 128,
    "long_bench/gov_report": 512,
    "long_bench/qmsum": 512,
    "long_bench/multi_news": 512,
    "long_bench/vcsum": 512,
    "long_bench/trec": 64,
    "long_bench/triviaqa": 32,
    "long_bench/samsum": 128,
    "long_bench/passage_count": 32,
    "long_bench/passage_retrieval_en": 32,
    "long_bench/passage_retrieval_zh": 32,
    "long_bench/lcc": 64,
    "long_bench/repobench-p": 64
}

DATADIR = {
    'ruler': 'data/ruler/data',
    'niah': 'data/niah/data',
}

Templates = {
    'base': "{ctx}",
    'llama-3': "<|start_header_id|>system<|end_header_id|>You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>{ctx}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    'yi': "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{ctx}<|im_end|>\n<|im_start|>assistant\n",
    'glm': "<|system|>\nYou are a helpful assistant\n<|user|> \n{ctx}<|assistant|>\n",
    'lwm': "You are a helpful assistant.\nUSER: {ctx}\nASSISTANT: Answer: ",
    'qwen': "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{ctx}<|im_end|>\n<|im_start|>assistant\n",
    'phi': "<|system|>\nYou are a helpful assistant<|end|>\n<|user|>\n{ctx}<|end|>\n<|assistant|>\n",
    "deepseek": "<｜begin▁of▁sentence｜>User: {task_template}\n\nAssistant:",
}


class Dataset:
    def __init__(self, dataset_name, tokenizer, datalen, num_samples, rank=0, world_size=1, inference_mode='single_turn'):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.datalen = datalen
        self.num_samples = num_samples
        self.rank = rank
        self.world_size = world_size
        self.is_sharded = False
        self.inference_mode = inference_mode
        self.tokenized_contexts = None
        self.tokenized_queries = None

        if dataset_name == 'niah':
            self.tokenized_prompts, self.gt, self.ctx_len, self.depth_pct = self.get_dataset()
        elif 'long_bench' in dataset_name:
            self.tokenized_prompts, self.gt, self.classes = self.get_dataset()
        else:
            self.tokenized_prompts, self.gt = self.get_dataset()

        self.num_samples = len(self.tokenized_prompts)
        self.gen_len = self.get_gen_len()
        self.metric = self.get_metric()

    def __str__(self) -> str:
        return f"Dataset: {self.dataset_name}, Num Samples: {self.num_samples}, Gen Len: {self.gen_len}, DataLen: {self.datalen}"

    def __repr__(self) -> str:
        return f"Dataset: {self.dataset_name}, Num Samples: {self.num_samples}, Gen Len: {self.gen_len}, DataLen: {self.datalen}"

    def __len__(self) -> int:
        return self.num_samples

    def shard(self, rank, world_size):
        if world_size > 1:
            shard_size = self.num_samples // world_size
            start = rank * shard_size
            end = start + shard_size if rank != world_size - 1 else self.num_samples
            shard_tokenized_prompts, shard_gt = self.tokenized_prompts[start:end], self.gt[start:end]
            self.tokenized_prompts = shard_tokenized_prompts
            self.gt = shard_gt
            if self.tokenized_contexts is not None:
                self.tokenized_contexts = self.tokenized_contexts[start:end]
            if self.tokenized_queries is not None:
                self.tokenized_queries = self.tokenized_queries[start:end]
            self.num_samples = len(shard_tokenized_prompts)

        self.is_sharded = True

    def get_gen_len(self):
        if 'niah' == self.dataset_name:
            return 10
        elif 'niah' in self.dataset_name:
            return 128
        elif 'vt' in self.dataset_name:
            return 30
        elif 'cwe' in self.dataset_name:
            return 120
        elif 'fwe' in self.dataset_name:
            return 50
        elif 'qa' in self.dataset_name:
            return 32
        elif 'long_bench' in self.dataset_name:
            return GEN_LEN[self.dataset_name]
        else:
            raise Exception("Gen len not found")

    def __getitem__(self, idx):
        if 'persona' in self.dataset_name:
            return self.tokenized_prompts[idx], self.queries[idx], self.gt[idx]
        return self.tokenized_prompts[idx], self.gt[idx]

    def get_metric(self):
        if 'long_bench' in self.dataset_name and self.dataset_name in METRICS_FN:
            return METRICS_FN[self.dataset_name]
        elif 'multiquery' in self.dataset_name or 'multivalue' in self.dataset_name:
            return METRICS_FN['multi']
        elif 'niah' in self.dataset_name:
            return METRICS_FN['niah']
        elif 'vt' in self.dataset_name:
            return METRICS_FN['vt']
        elif 'cwe' in self.dataset_name:
            return METRICS_FN['cwe']
        elif 'fwe' in self.dataset_name:
            return METRICS_FN['fwe']
        elif 'qa' in self.dataset_name:
            return METRICS_FN['qa']
        else:
            raise Exception("Metric not found")

    @staticmethod
    def _find_query_boundary(text):
        """Find the character index where the query starts in a ruler input.

        Searches for dataset-specific question markers near the end of the text.
        Returns the index of the newline before the query, or None if not found.
        """
        import re
        patterns = [
            r'\nWhat (?:is|are) (?:the |all the )?special magic number',
            r'\n+Question:',
            r'\n+Answer the question',
        ]
        best = -1
        for pattern in patterns:
            for m in re.finditer(pattern, text):
                best = max(best, m.start())
        return best if best > 0 else None

    def get_dataset(self):
        if 'ruler' in self.dataset_name: # ruler/xxx
            task = self.dataset_name.split('/')[-1]
            assert self.datalen in [2*1024, 4*1024, 8*1024, 16*1024, 32*1024, 64*1024, 128*1024, 256*1024, 512*1024, 600000], "Only support datalen of 2k, 4k, 8k, 16k, 32k, 64k, 128k, 256k, 512k, 600000"

            if 'llama-3' in self.tokenizer.name_or_path.lower():
                model_dir = 'llama-3'
            elif 'yi' in self.tokenizer.name_or_path.lower():
                model_dir = 'yi'
            elif 'lwm' in self.tokenizer.name_or_path.lower():
                model_dir = 'lwm'
            elif 'glm' in self.tokenizer.name_or_path.lower():
                model_dir = 'glm'
            elif 'qwen' in self.tokenizer.name_or_path.lower():
                model_dir = 'qwen'
            elif 'phi' in self.tokenizer.name_or_path.lower():
                model_dir = 'phi'
            elif 'deepseek' in self.tokenizer.name_or_path.lower():
                model_dir = 'deepseek'
            else:
                raise Exception("Model not found", self.tokenizer.name_or_path)

            dataset = load_dataset("json", data_files=f'{DATADIR["ruler"]}/{model_dir}/{self.datalen}/{task}/validation.jsonl', split='train')
            if self.num_samples > 0:
                self.num_samples = min(self.num_samples, len(dataset))
            else:
                self.num_samples = len(dataset)
            tokenized_prompts = []
            tokenized_contexts = []
            tokenized_queries = []
            gt = []

            if 'multiturn' in self.dataset_name:
                # Multi-turn format: input + queries[] + answers[]
                for i in range(self.num_samples):
                    input_text = dataset[i]['input']
                    first_query = dataset[i]['queries'][0]
                    combined_text = input_text + first_query
                    input_ids = self.tokenizer.encode(combined_text, return_tensors="pt", add_special_tokens=False)
                    tokenized_prompts.append(input_ids)

                    # Tokenize remaining queries individually
                    tokenized_query_list = []
                    for query in dataset[i]['queries'][1:]:
                        query_ids = self.tokenizer.encode(query, return_tensors="pt", add_special_tokens=False)
                        tokenized_query_list.append(query_ids)
                    tokenized_queries.append(tokenized_query_list)
                    gt.append(dataset[i]['answers'])

                self.tokenized_contexts = [None] * self.num_samples
                self.tokenized_queries = tokenized_queries
                return tokenized_prompts, gt
            else:
                for i in range(self.num_samples):
                    # Check for explicit context/query fields first (KeySifter approach)
                    if 'context' in dataset[i] and 'query' in dataset[i]:
                        context = dataset[i]['context']
                        query = dataset[i]['query']
                        input_text = context + query
                        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False)
                        tokenized_prompts.append(input_ids)
                        gt.append(dataset[i]['outputs'])

                        if self.inference_mode == 'multi_turn':
                            ctx_ids = self.tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
                            q_ids = self.tokenizer.encode(query, return_tensors="pt", add_special_tokens=False)
                            tokenized_contexts.append(ctx_ids)
                            tokenized_queries.append(q_ids)
                    else:
                        input_text = dataset[i]['input']
                        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False)
                        tokenized_prompts.append(input_ids)
                        gt.append(dataset[i]['outputs'])

                        if self.inference_mode == 'multi_turn':
                            boundary = self._find_query_boundary(input_text)
                            if boundary is not None:
                                context_text = input_text[:boundary]
                                query_text = input_text[boundary:]
                                ctx_ids = self.tokenizer.encode(context_text, return_tensors="pt", add_special_tokens=False)
                                q_ids = self.tokenizer.encode(query_text, return_tensors="pt", add_special_tokens=False)
                                tokenized_contexts.append(ctx_ids)
                                tokenized_queries.append(q_ids)
                            else:
                                print(colored(f"[Warning] Could not find query boundary for sample {i} in {self.dataset_name}, falling back to single_turn", 'red'))
                                tokenized_contexts.append(None)
                                tokenized_queries.append(None)

                if self.inference_mode == 'multi_turn':
                    self.tokenized_contexts = tokenized_contexts
                    self.tokenized_queries = tokenized_queries

                return tokenized_prompts, gt

        elif self.dataset_name == 'niah':
            print(colored(f"[Warning] NIAH dataset cannot set # samples, it is up to world_size, which is set to {self.world_size}", 'red'))
            
            haystack_file = f'{DATADIR["niah"]}/pg19_mini.jsonl'
            context_lengths_min = 16*1024
            context_lengths_max = self.datalen
            n_context_length_intervals = 15
            n_document_depth_intervals = 10  # position of the needle in the haystack
            n_rounds = 1 # max(1, 4 // self.world_size) # 8 rounds in total assume we have 8xGPUs
            needle = "\nThe special magic {city} number is: {rnd_number}\n"
            retrieval_question="What is the special magic {} number?"
            rnd_number_digits = 7

            context_lengths = np.round(
                np.linspace(
                    context_lengths_min,
                    context_lengths_max,
                    num=n_context_length_intervals,
                    endpoint=True,
                )
            ).astype(int)

            document_depth_percents = np.round( # we use linear scale here
                np.linspace(
                    0,
                    100,
                    num=n_document_depth_intervals,
                    endpoint=True,
                )
            ).astype(int)

            self.is_sharded = True # we shard the data during init dataset
            
            full_contexts = read_context_files(n=n_rounds, context_lengths=context_lengths, haystack_file=haystack_file, tokenizer=self.tokenizer)
            full_tokens = [
                self.tokenizer.encode(full_context, add_special_tokens=False) for full_context in full_contexts
            ]

            tokenized_prompts = []
            gt = []
            ctx_len = []
            depth_pct = []

            for context_length in context_lengths:
                trim_contexts = [
                    self.tokenizer.decode(full_token[:context_length], skip_special_tokens=True)
                    for full_token in full_tokens
                ]
                contexts = []
                for depth_percent in document_depth_percents:
                    for i in range(n_rounds):
                        random_city = random.choice(RANDOM_NEEDLE_CITIES)
                        insert_needle = True
                        needle_rnd_number = str(generate_random_number(rnd_number_digits))
                        context = create_contexts(
                            needle_rnd_number=needle_rnd_number,
                            insert_needle=insert_needle,
                            random_city=random_city,
                            trim_context=trim_contexts[i],
                            context_length=context_length,
                            depth_percent=depth_percent,
                            needle=needle,
                            retrieval_question=retrieval_question,
                            tokenizer=self.tokenizer,
                            final_context_length_buffer=32,
                        )
                        contexts.append(context)

                for context in contexts:
                    prompt = NIAH_TEMPLATE.format(
                        context=context["context"], question=context["question"]
                    )
                    input_tensor = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
                    tokenized_prompts.append(input_tensor.input_ids)
                    gt.append(context["needle_rnd_number"])
                    ctx_len.append(context["context_length"])
                    depth_pct.append(context["depth_percent"])
            
            return tokenized_prompts, gt, ctx_len, depth_pct

        elif 'long_bench' in self.dataset_name:
            task = self.dataset_name.split('/')[-1]
            dataset = load_dataset('THUDM/LongBench', task, split='test', trust_remote_code=True)
            use_chat_template = task not in ["lcc", "repobench-p", "samsum", "trec", "triviaqa"]

            if self.num_samples > 0:
                self.num_samples = min(self.num_samples, len(dataset))
            else:
                self.num_samples = len(dataset)
            tokenized_prompts = []
            gt = []
            classes = []

            for i in range(len(dataset)):
                if use_chat_template:
                    if 'llama-3' in self.tokenizer.name_or_path.lower():
                        model_template = Templates['llama-3'].format(ctx=LONG_BENCH_TEMPLATE[task])
                    elif 'yi' in self.tokenizer.name_or_path.lower():
                        model_template = Templates['yi'].format(ctx=LONG_BENCH_TEMPLATE[task])
                    elif 'glm' in self.tokenizer.name_or_path.lower():
                        model_template = Templates['glm'].format(ctx=LONG_BENCH_TEMPLATE[task])
                    elif 'qwen' in self.tokenizer.name_or_path.lower():
                        model_template = Templates['qwen'].format(ctx=LONG_BENCH_TEMPLATE[task])
                    elif 'phi' in self.tokenizer.name_or_path.lower():
                        model_template = Templates['phi'].format(ctx=LONG_BENCH_TEMPLATE[task])
                    elif "deepseek" in self.tokenizer.name_or_path.lower():
                        model_template = Templates['deepseek'].format(task_template=LONG_BENCH_TEMPLATE[task])
                    else:
                        raise Exception("Model not found for chat template", self.tokenizer.name_or_path)
                else:
                    model_template = LONG_BENCH_TEMPLATE[task]

                input_text = model_template.format(**dataset[i])
                # import pdb; pdb.set_trace()
                #breakpoint()
                # input_ids = truncate_by_tokens(input_text, self.tokenizer, self.datalen)
                input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

                if input_ids.shape[-1] <= self.datalen and input_ids.shape[-1] > 4096:
                    tokenized_prompts.append(input_ids)
                    gt.append(dataset[i]['answers'])
                    classes.append(dataset[i]['all_classes'])

            return tokenized_prompts, gt, classes

        else:
            raise ValueError(f"Dataset {self.dataset_name} not found, please choose in ruler, persona, infini_bench, needle, niah, long_bench")