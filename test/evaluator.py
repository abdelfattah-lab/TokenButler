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

import os
import torch
from termcolor import colored
from tqdm import tqdm
import torch.distributed as dist
import pandas as pd
import json
import datetime

from data.dataset import Dataset
from models.base import LLM


class Evaluator:
    def __init__(self, dist_config):

        self.dist_config = dist_config

        # init final report
        self.all_stats = []

    def test(self, llm: LLM, dataset: Dataset, output_path: str, setting: str = 'baseline'):

        # mkdir if not exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if self.dist_config.master_process:
            print(colored(f"[Test] {llm.model_name} on {dataset.dataset_name}, results saved to {output_path}", 'green'))

        if dataset.is_sharded == False:
            dataset.shard(self.dist_config.rank, self.dist_config.world_size)

        bsz = llm.batch_size
        scores = []
        preds = []

        # clear the file
        open(output_path, 'w').close()
        if self.dist_config.is_distributed:
            dist.barrier()

        is_multi_turn_ruler = (
            getattr(dataset, 'inference_mode', 'single_turn') == 'multi_turn'
            and 'ruler' in dataset.dataset_name
            and dataset.tokenized_contexts is not None
        )

        progress_bar = tqdm(range(dataset.num_samples // bsz), desc='Testing', disable=self.dist_config.is_distributed and not self.dist_config.master_process)
        for i in range(dataset.num_samples // bsz):
            prompt = torch.cat([dataset.tokenized_prompts[i*bsz+j] for j in range(bsz)], dim=0)
            if is_multi_turn_ruler:
                assert bsz == 1, "Multi-turn ruler evaluation requires batch_size=1"
                context = dataset.tokenized_contexts[i]
                query = dataset.tokenized_queries[i]
                if context is not None and query is not None:
                    # Turn 1: prefill context only (no extra tokens generated)
                    llm.prefill(context.to(llm.device))
                    llm.kv_cache.H2D()
                    # Turn 2: generate from query using cached context
                    rets = llm.generate(query.to(llm.device), cont=True, gen_len=dataset.gen_len, top_p=1.0, temperature=0.0)
                else:
                    # Fallback to single-turn if split failed
                    rets = llm.generate(prompt.to(llm.device), gen_len=dataset.gen_len, verbose=False, top_p=1.0, temperature=0.0)
                for (pred, gt) in zip(rets, dataset.gt[i*bsz:(i+1)*bsz]):
                    if isinstance(gt, list):
                        if len(gt) == 1:
                            gt = gt[0]
                    scores.append(dataset.metric(pred, gt))

            elif 'persona' in dataset.dataset_name:
                assert bsz == 1
                llm.generate(prompt.to(llm.device), gen_len=0, verbose=False, top_p=0.9, temperature=0.0) # prefill ctx
                queries = dataset.queries[i]
                gts_list = dataset.gt[i]
                rets_list = []
                for query, gts in zip(queries, gts_list):
                    rets = llm.generate(llm.encode(query, template="chat"), cont=True, gen_len=dataset.gen_len, top_p=0.9, temperature=0.0)
                    rets_list.append(rets)
                    pred = rets[0]
                    if isinstance(gts, list):
                        if len(gts) == 1:
                            gts = gts[0]
                    # print(pred, gts, dataset.metric(pred, gts))
                    scores.append(dataset.metric(pred, gts))
                
            elif 'long_bench' in dataset.dataset_name:
                rets = llm.generate(prompt.to(llm.device), gen_len=dataset.gen_len, verbose=False, top_p=1.0, temperature=0.0)
                for (pred, gt, all_classes) in zip(rets, dataset.gt[i*bsz:(i+1)*bsz], dataset.classes[i*bsz:(i+1)*bsz]):
                    scores.append(max([dataset.metric(pred, g, all_classes=all_classes) for g in gt]))

            else:
                rets = llm.generate(prompt.to(llm.device), gen_len=dataset.gen_len, verbose=False, top_p=1.0, temperature=0.0)
                for (pred, gt) in zip(rets, dataset.gt[i*bsz:(i+1)*bsz]):
                    if isinstance(gt, list):
                        if len(gt) == 1:
                            gt = gt[0]
                    scores.append(dataset.metric(pred, gt))
            
            progress_bar.update(1)
            avg_score = sum(scores) / len(scores)
            progress_bar.set_postfix({'avg_score': avg_score, 'max_mem': torch.cuda.max_memory_allocated(0) / 1024**3})

            if dataset.dataset_name == 'niah':
                preds = {
                        "context_length": dataset.ctx_len[i*bsz:(i+1)*bsz],
                        "depth_percent": dataset.depth_pct[i*bsz:(i+1)*bsz],
                        "response": rets,
                        "answer": dataset.gt[i*bsz:(i+1)*bsz],
                        "correct": scores[i*bsz:(i+1)*bsz],
                        "avg_score": avg_score,
                    }
            elif 'persona' in dataset.dataset_name:
                preds = {
                        "query": queries,
                        "response": rets_list,
                        "answer": dataset.gt[i*bsz:(i+1)*bsz],
                        "correct": scores,
                        "avg_score": avg_score,
                    }
            else:
                preds = {
                        "prediction": rets,
                        "ground_truth": dataset.gt[i*bsz:(i+1)*bsz],
                        "correct": scores,
                        "avg_score": avg_score,
                    }

            with open(output_path, "a", encoding="utf8") as fout:
                fout.write(json.dumps(preds, ensure_ascii=False) + "\n")
            # if self.dist_config.is_distributed:
            #     dist.barrier()

        progress_bar.close()
        avg_score = sum(scores) / len(scores)

        self.all_stats.append(
            {
                'model': llm.model_name,
                'dataset': dataset.dataset_name,
                'samples': dataset.num_samples,
                f'{setting}': avg_score,
            }
        )
        if self.dist_config.is_distributed:
            dist.barrier()

    def summarize(self):
        df = pd.DataFrame(self.all_stats)

        if self.dist_config.is_distributed:
            dist.barrier()
            output = [None for _ in range(self.dist_config.world_size)]
            dist.gather_object(df, output if self.dist_config.master_process else None, dst=0)
            dist.barrier()
            if self.dist_config.master_process:
                df = pd.concat(output)
                # Define the columns you want to calculate the mean for (excluding 'samples')
                setting_columns = [col for col in df.columns if col not in ['model', 'dataset', 'samples']]

                # Calculate the sum for 'samples'
                samples_sum = df.groupby(['model', 'dataset'])['samples'].sum()

                # Define the aggregation dictionary for other settings
                agg_dict = {col: 'mean' for col in setting_columns}

                # Calculate the weighted mean for each setting column based on 'samples'
                weighted_means = df.groupby(['model', 'dataset']).apply(lambda x: pd.Series({
                    col: (x[col] * x['samples']).sum() / x['samples'].sum() for col in setting_columns
                }))

                # Combine the weighted means with the sum of samples
                df = weighted_means.join(samples_sum).reset_index()
        
        if self.dist_config.master_process:
            # add a row for the average
            numeric_columns = df.select_dtypes(include='number')
            mean_values = numeric_columns.mean()
            mean_row = pd.DataFrame({col: [mean_values[col] if col in mean_values else 'mean'] for col in df.columns})
            df_with_mean = pd.concat([df, mean_row], ignore_index=True)
            print(df_with_mean.to_markdown(index=False))