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

# Base LLM class

import torch
import os
import torch.nn.functional as F
import time
import gc
from tqdm import tqdm
from contextlib import nullcontext

from flash_attn import flash_attn_with_kvcache
import torch.cuda.nvtx as nvtx
from torch.profiler import profile, record_function, ProfilerActivity

from .tensor_op import sample_token, layer_norm, minference_prefill_kernel
from .kv_cache import KV_Cache, ShadowKVCache, ShadowKVCache_CPU
from .kv_cache_xkv import ShadowKVCache_xKey, ShadowKVCache_xKV, ShadowKVCache_xKey_CPU, ShadowKVCache_xKV_CPU
from .kv_cache_keysifter import KeySifterCache
from .kv_cache_keysifter_cpu import KeySifterCache_CPU
from .kv_cache_cpu import KV_Cache_CPU
from .kv_cache_oracle import OracleCache
from .kv_cache_oracle_cpu import OracleCache_CPU
from .kv_cache_dsa import DSACache
from .merge_configs import xKVConfig

class LLM:

    def __str__(self) -> str:
        gpu_mem = f"{round(torch.cuda.memory_allocated(self.device) / 1024**3, 2)} GB / {round(torch.cuda.get_device_properties(self.device).total_memory / 1024**3, 2)} GB"
        return f"LLM: {self.model_name}, attn_mode: {self.attn_mode}, max_length: {self.max_length}, batch_size: {self.batch_size}, device: {self.device}, dtype: {self.dtype}, GPU mem: {gpu_mem}"
    
    def _maybe_record_function(self, name):
        """Return record_function context if profiling is enabled, otherwise nullcontext"""
        return record_function(name) if getattr(self, '_profiling_enabled', False) else nullcontext()
    
    def set_prefill_chunk_size(self, chunk_size: int):
        """Set the chunk size for chunked prefill (to handle very long contexts without OOM).
        
        Args:
            chunk_size: Maximum tokens to process in a single prefill pass. 
                       Longer sequences will be split into chunks.
                       Set to None or 0 to disable chunking.
        """
        self.prefill_chunk_size = chunk_size if chunk_size and chunk_size > 0 else None

    def init_kv_cache(self, sparse_budget: int, chunk_size: int, config, rank: int, merge_config: xKVConfig, keysifter_predictor=None, producer_frequency: int = 4, dDash: int = 16, oracle_random_indices: bool = True, page_size: int = 1, local_window: int = 512, min_sparse_index: int = 128, quantize_int8: bool = False, cpu_chunk_size: int = 4096, predict_interval: int = 1, enable_neighbor_fetch: bool = False):
        if self.attn_mode == 'full':
            self.kv_cache = KV_Cache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size)
        elif self.attn_mode.lower() == 'shadowkv':
            self.kv_cache = ShadowKVCache(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, chunk_size=chunk_size, rank=rank)
        elif self.attn_mode.lower() == 'shadowkv_cpu':
            self.kv_cache = ShadowKVCache_CPU(config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, chunk_size=chunk_size, rank=rank)
        elif self.attn_mode.lower() == 'shadowkv_xkey':
            self.kv_cache = ShadowKVCache_xKey(config, merge_config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, chunk_size=chunk_size)
        elif self.attn_mode.lower() == 'shadowkv_xkv':
            self.kv_cache = ShadowKVCache_xKV(config, merge_config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, chunk_size=chunk_size)
        elif self.attn_mode.lower() == 'shadowkv_xkey_cpu':
            self.kv_cache = ShadowKVCache_xKey_CPU(config, merge_config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, chunk_size=chunk_size)
        elif self.attn_mode.lower() == 'shadowkv_xkv_cpu':
            self.kv_cache = ShadowKVCache_xKV_CPU(config, merge_config, max_length=self.max_length, device=self.device, dtype=self.dtype, batch_size=self.batch_size, sparse_budget=sparse_budget, chunk_size=chunk_size)
        elif self.attn_mode.lower() == 'keysifter':
            if keysifter_predictor is None:
                raise ValueError("KeySifter mode requires a predictor. Pass keysifter_predictor to init_kv_cache.")
            self.kv_cache = KeySifterCache(
                config,
                predictor=keysifter_predictor,
                max_length=self.max_length,
                device=self.device,
                dtype=self.dtype,
                batch_size=self.batch_size,
                sparse_budget=sparse_budget,
                chunk_size=chunk_size,
                producer_frequency=producer_frequency,
                local_window=local_window,
                min_sparse_index=min_sparse_index,
                quantize_int8=quantize_int8,
                predict_interval=predict_interval,
                enable_neighbor_fetch=enable_neighbor_fetch,
            )
        elif self.attn_mode.lower() == 'dsa':
            from models.kv_cache_dsa import DSACache, LightningIndexer
            dsa_indexer = LightningIndexer(
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                producer_frequency=producer_frequency,
                device=self.device,
                dtype=self.dtype,
            )
            self.kv_cache = DSACache(
                config,
                indexer=dsa_indexer,
                max_length=self.max_length,
                device=self.device,
                dtype=self.dtype,
                batch_size=self.batch_size,
                sparse_budget=sparse_budget,
                chunk_size=chunk_size,
                producer_frequency=producer_frequency,
                local_window=local_window,
                min_sparse_index=min_sparse_index,
                predict_interval=predict_interval,
                enable_neighbor_fetch=enable_neighbor_fetch,
            )
        elif self.attn_mode.lower() == 'oracle':
            self.kv_cache = OracleCache(
                config,
                batch_size=self.batch_size,
                max_length=self.max_length,
                device=self.device,
                dtype=self.dtype,
                sparse_budget=sparse_budget,
                chunk_size=chunk_size,
                random_indices=oracle_random_indices,
                page_size=page_size,
                predict_interval=predict_interval,
            )
        elif self.attn_mode.lower() == 'keysifter_cpu':
            if keysifter_predictor is None:
                raise ValueError("KeySifter CPU mode requires a predictor. Pass keysifter_predictor to init_kv_cache.")
            self.kv_cache = KeySifterCache_CPU(
                config,
                predictor=keysifter_predictor,
                max_length=self.max_length,
                device=self.device,
                dtype=self.dtype,
                batch_size=self.batch_size,
                sparse_budget=sparse_budget,
                chunk_size=chunk_size,
                producer_frequency=producer_frequency,
                local_window=local_window,
                min_sparse_index=min_sparse_index,
                quantize_int8=quantize_int8,
                predict_interval=predict_interval,
                enable_neighbor_fetch=enable_neighbor_fetch,
            )
        elif self.attn_mode.lower() == 'oracle_cpu':
            self.kv_cache = OracleCache_CPU(
                config,
                batch_size=self.batch_size,
                max_length=self.max_length,
                device=self.device,
                dtype=self.dtype,
                sparse_budget=sparse_budget,
                chunk_size=chunk_size,
                producer_frequency=producer_frequency,
                local_window=local_window,
                min_sparse_index=min_sparse_index,
                random_indices=oracle_random_indices,
                page_size=page_size,
                predict_interval=predict_interval,
            )
        elif self.attn_mode.lower() == 'full_cpu':
            self.kv_cache = KV_Cache_CPU(
                config,
                batch_size=self.batch_size,
                max_length=self.max_length,
                device=self.device,
                dtype=self.dtype,
                chunk_size=cpu_chunk_size,
            )
        else:
            raise ValueError(f"Invalid attention mode {self.attn_mode}")

    def print_kv_stats(self):
        self.kv_cache.print_stats()
    
    def get_ctx(self, input_ids: torch.LongTensor):
        """
        Returns position ids for the current input based on the kv cache length.
        
        Args:
            input_ids (torch.LongTensor): The input token IDs.
        """
        input_len = input_ids.size(1)
        past_len = self.kv_cache.get_kv_len()
        position_ids = torch.arange(past_len, past_len + input_len, device=self.device, dtype=torch.long).unsqueeze(0).repeat(input_ids.size(0), 1)
        return position_ids

    @torch.inference_mode()
    def inference(self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor):

        hidden_states = F.embedding(input_ids, self.embed_tokens)

        for idx in range(self.num_layers):
            hidden_states = self.layer_compute(self.layers[idx], idx, hidden_states, position_ids)
        hidden_states = layer_norm(hidden_states, w=self.norm_weight, eps=self.norm_variance_epsilon)
        
        if hidden_states.shape[1] > 16: # prefill
            hidden_states = hidden_states[:, -1:, :]
        logits = F.linear(hidden_states, self.lm_head).float()
        
        return logits

    @torch.inference_mode()
    def prefill(self, input_ids: torch.LongTensor):
        self.kv_cache.clear()
        
        # Check if we need to chunk the prefill
        seq_len = input_ids.shape[1]
        chunk_size = getattr(self, 'prefill_chunk_size', None)
        
        if chunk_size is not None and seq_len > chunk_size:
            # Chunked prefill: process the prompt in chunks to avoid OOM on very long contexts
            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            print(f"[Chunked Prefill] Processing {seq_len} tokens in {num_chunks} chunks of {chunk_size} tokens")
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, seq_len)
                chunk_ids = input_ids[:, start_idx:end_idx]
                
                # Get position ids for this chunk
                position_ids = self.get_ctx(chunk_ids)
                
                # Process chunk (accumulates KV cache)
                logits = self.inference(input_ids=chunk_ids, position_ids=position_ids)
                
                # Free memory after each chunk
                if i < num_chunks - 1:  # Don't clear on last chunk
                    torch.cuda.empty_cache()
        else:
            # Normal prefill: process entire sequence at once
            logits = self.inference(input_ids=input_ids, position_ids=self.get_ctx(input_ids))

        assert self.kv_cache.get_kv_len() == input_ids.shape[-1], f"KV length mismatch, got {self.kv_cache.get_kv_len()}, expected {input_ids.shape[-1]}"
        return logits

    @torch.inference_mode()
    def prefill_cont(self, input_ids: torch.LongTensor):
        if isinstance(self.kv_cache, (KeySifterCache, KeySifterCache_CPU, OracleCache, OracleCache_CPU, DSACache)):
            # Process tokens one by one to maintain causality.
            # The decode path's update_kv_cache + flash_attn_with_kvcache assumes single-token input;
            # batching multiple tokens causes q[0] to attend to future tokens via k_cache.
            #
            # During continuation (new query), optionally override predict_interval
            # to 1 so every query token gets a fresh sparse selection.
            # When prefill_cont_dense=True (default), interval is forced to 1.
            # When prefill_cont_dense=False, the original interval is kept.
            saved_interval = None
            if getattr(self.kv_cache, 'prefill_cont_dense', True):
                saved_interval = self.kv_cache.predict_interval
                self.kv_cache.predict_interval = 1
            seq_len = input_ids.size(1)
            for t in range(seq_len):
                token = input_ids[:, t:t+1]
                logits = self.inference(input_ids=token, position_ids=self.get_ctx(token))
            # Restore the interval for the upcoming decode (answer generation).
            if saved_interval is not None:
                self.kv_cache.predict_interval = saved_interval
            # Force a fresh prediction on the first real generated token so it
            # starts from accurate importance scores regardless of predict_interval.
            if hasattr(self.kv_cache, '_force_next_prediction'):
                self.kv_cache._force_next_prediction = True
            return logits
        else:
            logits = self.inference(input_ids=input_ids, position_ids=self.get_ctx(input_ids))
            return logits
    
    def encode(self, text: str, template=None, truncation=False):
        if template == 'chat':
            text = self.chat_template.format(msg=text)
            input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            if self.tokenizer.bos_token_id is not None:
                assert self.tokenizer.bos_token_id not in input_ids, f"bos_token_id found in input_ids"
            return input_ids
        if template == 'ctx':
            text = self.ctx_template.format(ctx=text)
        if template == 'prefix':
            text = self.prefix_template.format(ctx=text)
        input_ids = self.tokenizer(text, return_tensors="pt", truncation=truncation).input_ids.to(self.device)
        return input_ids

    @torch.inference_mode()
    def layer_compute(self, 
            buffer,
            layer_idx :int, 
            hidden_states: torch.FloatTensor, 
            position_ids: torch.LongTensor):

        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = self.pre_attention_compute(
            hidden_states,
            buffer,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim
        )
        
        if isinstance(self.kv_cache, KV_Cache):
            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
            key_states, value_states = self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)
            
            if self.minference == True and q_len > 1:
                hidden_states = minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
            else:
                hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

        elif isinstance(self.kv_cache, ShadowKVCache) or isinstance(self.kv_cache, ShadowKVCache_CPU) or \
             isinstance(self.kv_cache, ShadowKVCache_xKey) or isinstance(self.kv_cache, ShadowKVCache_xKV) or \
             isinstance(self.kv_cache, ShadowKVCache_xKey_CPU) or isinstance(self.kv_cache, ShadowKVCache_xKV_CPU) or \
             isinstance(self.kv_cache, KeySifterCache) or isinstance(self.kv_cache, KeySifterCache_CPU) or \
             isinstance(self.kv_cache, OracleCache) or isinstance(self.kv_cache, OracleCache_CPU) or \
             isinstance(self.kv_cache, DSACache):

            # Prefill: use for long sequences OR first pass of short sequences with KeySifter/Oracle/DSA
            # NOTE: When using chunked prefill, each chunk goes through this prefill path separately.
            # The decode path below (q_len == 1) is completely unaffected by chunking.
            is_keysifter_first_pass = (isinstance(self.kv_cache, (KeySifterCache, KeySifterCache_CPU, OracleCache, OracleCache_CPU, DSACache))) and self.kv_cache.prefill_len == 0
            if q_len > 1024 or is_keysifter_first_pass: # prefill
                with self._maybe_record_function("batch_prefill"):
                    # svd unrope key and save
                    if isinstance(self.kv_cache, ShadowKVCache_xKey):
                        self.kv_cache.get_svd(key_states, layer_idx, fake_svd=self.fake_svd)
                    elif isinstance(self.kv_cache, ShadowKVCache_xKV):
                        self.kv_cache.get_svd(key_states, value_states, layer_idx, fake_svd=self.fake_svd)
                    elif isinstance(self.kv_cache, ShadowKVCache_xKV_CPU):
                        self.kv_cache.get_svd(key_states, value_states, layer_idx)
                    elif isinstance(self.kv_cache, (KeySifterCache, KeySifterCache_CPU, OracleCache, OracleCache_CPU, DSACache)):
                        # Store un-RoPEd keys and compute projections of keys for importance scoring (skipped for Oracle/DSA)
                        self.kv_cache.get_svd(key_states, layer_idx)
                    else:
                        self.kv_cache.get_svd(key_states, layer_idx)
                    query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
                    self.kv_cache.prefill_kv_cache(value_states, layer_idx, key_states, query_states[:, :, -1:])
                    
                    if self.minference == True:
                        hidden_states = minference_prefill_kernel(query_states=query_states, key_states=key_states, value_states=value_states, minference_parttern=self.minference_parttern[layer_idx])
                    else:
                        hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)
            else: # decode

                # rope query and key
                with self._maybe_record_function("rope_query_key"):
                    query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)

                # update kv cache to buffer
                with self._maybe_record_function("update_kv_cache"):
                    self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)

                # KeySifter: compute importance queries and refetch for layer group at producer layers
                # Note: layer 0 also needs to run predictor for layers 1-3, so don't skip it here
                if isinstance(self.kv_cache, DSACache):
                    # DSA: run indexer at every layer (faithful to original DSA paper)
                    with self._maybe_record_function("dsa_prefetch"):
                        self.kv_cache.prefetch_single_layer(residual, layer_idx)
                elif isinstance(self.kv_cache, (KeySifterCache, KeySifterCache_CPU)):
                    producer_frequency = self.kv_cache.producer_frequency
                    if layer_idx % producer_frequency == 0:
                        with self._maybe_record_function("keysifter_prefetch"):
                            # hidden_states here is the residual (pre-attention), we need to pass it
                            self.kv_cache.prefetch_layer_group(residual, layer_idx)
                elif isinstance(self.kv_cache, OracleCache) or isinstance(self.kv_cache, OracleCache_CPU):
                    producer_frequency = self.kv_cache.producer_frequency
                    if layer_idx % producer_frequency == 0:
                        with self._maybe_record_function("oracle_prefetch"):
                            # Oracle prefetch
                            self.kv_cache.prefetch_layer_group(residual, layer_idx)

                # get retrieval idx
                with self._maybe_record_function("get_retrieval_position_ids"):
                    position_ids = self.kv_cache.get_retrieval_position_ids(layer_idx=layer_idx, query_states=query_states)

                # multi-stream
                if not isinstance(self.kv_cache, (ShadowKVCache_xKV_CPU, KeySifterCache, KeySifterCache_CPU, OracleCache, OracleCache_CPU, DSACache)):
                    curr_stream = torch.cuda.current_stream()
                    get_value_stream = self.kv_cache.copy_stream

                if isinstance(self.kv_cache, ShadowKVCache_xKV_CPU):
                    with self._maybe_record_function("get_value_cache_xKV_cpu"):
                        value_states = self.kv_cache.get_value_cache(layer_idx, position_ids, self.cos_sin_cache)
                elif isinstance(self.kv_cache, (KeySifterCache, KeySifterCache_CPU, OracleCache, OracleCache_CPU, DSACache)):
                    with self._maybe_record_function("get_value_cache_keysifter_or_oracle"):
                        value_states = self.kv_cache.get_value_cache(layer_idx, position_ids)
                else:
                    with self._maybe_record_function("get_value_cache_offload_stream"):
                        with torch.cuda.stream(get_value_stream):
                            get_value_stream.wait_stream(curr_stream)
                            value_states = self.kv_cache.get_value_cache(layer_idx, position_ids)

                # gather key cache from GPU and RoPE it (should be hide by CPU offloading time)
                if isinstance(self.kv_cache, ShadowKVCache_CPU) or isinstance(self.kv_cache, ShadowKVCache_xKey_CPU) or isinstance(self.kv_cache, ShadowKVCache_xKV_CPU):
                    with self._maybe_record_function("get_key_cache"):
                        key_states = self.kv_cache.get_key_cache(layer_idx=layer_idx, position_ids=position_ids, rope_func=self.apply_rotary_pos_emb_single, cos_sin_cache=self.cos_sin_cache)
                elif isinstance(self.kv_cache, (KeySifterCache, KeySifterCache_CPU, OracleCache, OracleCache_CPU, DSACache)):
                    with self._maybe_record_function("get_key_cache_keysifter_or_oracle"):
                        key_states = self.kv_cache.get_key_cache(layer_idx=layer_idx, position_ids=position_ids, rope_func=self.apply_rotary_pos_emb_single)
                else:
                    with self._maybe_record_function("get_key_cache"):
                        key_states = self.kv_cache.get_key_cache(layer_idx=layer_idx, position_ids=position_ids, rope_func=self.apply_rotary_pos_emb_single)

                if isinstance(self.kv_cache, ShadowKVCache_CPU) or isinstance(self.kv_cache, ShadowKVCache_xKey_CPU):
                    with self._maybe_record_function("wait_get_value_stream"):
                        curr_stream.wait_stream(get_value_stream)

                # flash attention
                with self._maybe_record_function("flash_attn_with_kvcache"):
                    hidden_states = flash_attn_with_kvcache(q=query_states.transpose(1, 2), k_cache=key_states.transpose(1, 2), v_cache=value_states.transpose(1, 2), causal=True)

        elif isinstance(self.kv_cache, KV_Cache_CPU):
            # Dense attention with CPU offloading - uses chunked streaming
            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)
            
            if q_len > 1:  # Prefill
                # Store to CPU cache
                self.kv_cache.prefill_kv_cache(value_states, layer_idx, key_states)
                # For prefill, compute attention with flash_attn using GPU tensors
                hidden_states = flash_attn_with_kvcache(
                    q=query_states.transpose(1, 2), 
                    k_cache=key_states.transpose(1, 2), 
                    v_cache=value_states.transpose(1, 2), 
                    causal=True
                )
            else:  # Decode
                # Update CPU cache with new token
                total_kv_len, _ = self.kv_cache.update_kv_cache(key_states, value_states, layer_idx)
                
                # Use chunked attention for long contexts
                # query_states: [bsz, num_heads, 1, head_dim]
                hidden_states = self.kv_cache.compute_chunked_attention(
                    query=query_states,
                    layer_idx=layer_idx,
                    total_kv_len=total_kv_len,
                )
                # Reshape from [bsz, num_heads, 1, head_dim] to [bsz, 1, num_heads, head_dim]
                hidden_states = hidden_states.transpose(1, 2)

        else:
            raise ValueError(f"Invalid attention mode {self.attn_mode}")

        hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)
        
        if bsz*q_len > 64*1024: # [bsz, seq, 128]
            output = torch.empty_like(hidden_states)
            prop_iter = bsz * q_len // (8*1024)
            prefill_chunk_size = bsz * q_len // prop_iter
            prefill_iter = (q_len + prefill_chunk_size - 1) // prefill_chunk_size
            for i in range(prefill_iter):
                start = i*prefill_chunk_size
                end = (i+1)*prefill_chunk_size
                output[:, start:end] = self.post_attention_compute(hidden_states[:, start:end], residual[:, start:end], buffer)

            hidden_states = output

        else:
            hidden_states = self.post_attention_compute(hidden_states, residual, buffer)
        
        return hidden_states

    def decode(self, input_ids: torch.Tensor, skip_special_tokens: bool = False):
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=skip_special_tokens)

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, gen_len: int = 256, temperature: float = 0.0, top_p: float = 0.9, top_k :int = 50, verbose: bool = False, benchmark: bool = False, cont: bool = False, enable_profiler: bool = False, profiler_output_dir: str = "./profiler_logs", profiler_wait_steps: int = 2, profiler_warmup_steps: int = 2, profiler_active_steps: int = 6):
        """accuracy eval usage, not for throughput eval
        
        Args:
            enable_profiler: Enable torch profiler to trace the inference workload
            profiler_output_dir: Directory to save profiler traces (default: ./profiler_logs)
            profiler_wait_steps: Number of steps to wait before profiling (default: 2)
            profiler_warmup_steps: Number of warmup steps before active profiling (default: 2)
            profiler_active_steps: Number of steps to actively profile (default: 6)
        """
        assert type(input_ids) == torch.Tensor, f"input_ids must be a torch.Tensor, got {type(input_ids)}"

        # prefill
        if cont == False:
            if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.prefill(input_ids)
        else:
            if input_ids.size(1) + self.kv_cache.get_kv_len() >= self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.prefill_cont(input_ids)
        next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        
        n = 0
        pos = 0
        generated_ids = []
        generated_ids.extend(next_token[0].tolist())
        
        self.kv_cache.H2D()

        if benchmark == True:
            start = time.time()
        
        # Profiler configuration - start after prefill (skip prefill to reduce file size)
        if enable_profiler:
            self._profiling_enabled = True
            os.makedirs(profiler_output_dir, exist_ok=True)
            
            profiler_schedule = torch.profiler.schedule(
                wait=profiler_wait_steps,
                warmup=profiler_warmup_steps,
                active=profiler_active_steps,
                repeat=1
            )
            
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=profiler_schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_output_dir),
                record_shapes=True,
                profile_memory=False,  # Disable memory profiling
                with_stack=True
            )
            prof.start()
            print(f"\nProfiler enabled: will profile {profiler_active_steps} decode steps after {profiler_wait_steps + profiler_warmup_steps} warmup (skipping prefill)")

        
        while n < gen_len:
            with self._maybe_record_function("decode_step"):
                logits = self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))
                next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
            
            n += 1
            generated_ids.extend(next_token[0].tolist())
            
            # Step profiler
            if enable_profiler:
                prof.step()
                # Stop profiler after wait + warmup + active steps
                if n == profiler_wait_steps + profiler_warmup_steps + profiler_active_steps:
                    self._profiling_enabled = False
                    prof.stop()
                    print(f"\nProfiler stopped after {profiler_active_steps} active decode steps")
                    print(f"Traces saved to: {profiler_output_dir}")
                    print(f"View with: tensorboard --logdir={profiler_output_dir}")
                    enable_profiler = False
            
            if verbose == True:
                generated_text = (
                    self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                        spaces_between_special_tokens=False,
                    ).strip().split(" ")
                )
                now = len(generated_text) - 1
                if now > pos:
                    print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                    pos = now

            if next_token[0] == self.tokenizer.eos_token_id:
                break
            if self.tokenizer.decode(next_token[0]) == "<|eot_id|>": # llama-3
                break
            if self.tokenizer.decode(next_token[0]) == "<|im_end|>": # yi
                break
            if next_token[0] in [151329, 151336, 151338]: # glm
                break
            if self.tokenizer.decode(next_token[0]) == "<|endoftext|>": # glm
                break
            if self.tokenizer.decode(next_token[0]) == "<|end|>": # phi
                break

        if verbose == True and n!=0:
            print(" ".join(generated_text[pos:]), end=" ", flush=True)
        if benchmark == True:
            end = time.time()
            print(f"\nPrefill {input_ids.size(1)} tokens | Generate {n} tokens in {round(end - start, 2)}s, {round(n / (end - start), 2)} tokens/s | cached {self.kv_cache.get_kv_len()}\n")

        # feed new token to the model
        self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))

        # TODO(max410011): Uncomment these lines during memory usage evaluation
        # gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()

        return [self.tokenizer.decode(generated_ids, skip_special_tokens=True)]
    
    @torch.inference_mode()
    def batch_prefill(self, input_ids: torch.Tensor, benchmark: bool = False):
        self.kv_cache.clear()
        batch_size = input_ids.size(0)
        
        assert batch_size == self.batch_size, f"batch_size mismatch, got {batch_size}, expected {self.batch_size}"
        
        if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
        
        logits = torch.zeros(batch_size, 1, self.vocab_size, device=self.device, dtype=torch.float32)

        if input_ids.shape[-1] > 120*1024 and input_ids.shape[-1] < 200*1024:
            T = 8
        else:
            T = 4
        # for bsz in range(0, batch_size, T):
        for bsz in tqdm(range(0, batch_size, T), desc=f"Prefilling (batch size={batch_size})"):
            req_input_ids = input_ids[bsz:bsz+T]
            logits[bsz:bsz+T].copy_(self.inference(input_ids=req_input_ids, position_ids=self.get_ctx(req_input_ids)))
        assert self.kv_cache.get_kv_len() == input_ids.shape[-1], f"KV length mismatch, got {self.kv_cache.get_kv_len()}, expected {input_ids.shape[-1]}"

        return logits


    @torch.inference_mode()
    def warmup(self):

        a = torch.randn(self.batch_size, 1024, 1024).to(self.dtype).to(self.device)
        b = torch.randn(self.batch_size, 1024, 1024).to(self.dtype).to(self.device)
        for _ in range(100):
            torch.bmm(a, b)
        del a, b

        # print("Warmup done")

    @torch.inference_mode()
    def batch_generate(self, input_ids: torch.Tensor, gen_len: int = 256, temperature: float = 0.0, top_p: float = -1, top_k :int = 50, verbose: bool = False, benchmark: bool = False, cont: bool = False, enable_profiler: bool = False, profiler_output_dir: str = "./profiler_logs", profiler_wait_steps: int = 2, profiler_warmup_steps: int = 2, profiler_active_steps: int = 6):
        """throughput eval usage
        
        Args:
            enable_profiler: Enable torch profiler to trace the inference workload
            profiler_output_dir: Directory to save profiler traces (default: ./profiler_logs)
            profiler_wait_steps: Number of steps to wait before profiling (default: 2)
            profiler_warmup_steps: Number of warmup steps before active profiling (default: 2)
            profiler_active_steps: Number of steps to actively profile (default: 6)
        """
        assert type(input_ids) == torch.Tensor, f"input_ids must be a torch.Tensor, got {type(input_ids)}"

        # prefill
        if cont == False:
            if input_ids.size(1) > self.max_length:
                raise ValueError(f"Input length must be less than {self.max_length}, but got {input_ids.size(1)}")
            logits = self.batch_prefill(input_ids)
        else:
            logits = self.prefill_cont(input_ids)
        next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
        
        n = 0
        generated_ids = []
        generated_ids.append(next_token[:, -1].tolist())
        
        self.kv_cache.H2D()
        self.warmup()

        if benchmark == True:
            start = time.time()
        
        # Profiler configuration - start after prefill (skip prefill to reduce file size)
        if enable_profiler:
            self._profiling_enabled = True
            os.makedirs(profiler_output_dir, exist_ok=True)
            
            profiler_schedule = torch.profiler.schedule(
                wait=profiler_wait_steps,
                warmup=profiler_warmup_steps,
                active=profiler_active_steps,
                repeat=1
            )
            
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=profiler_schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_output_dir),
                record_shapes=True,
                profile_memory=False,  # Disable memory profiling
                with_stack=True
            )
            prof.start()
            print(f"\nProfiler enabled: will profile {profiler_active_steps} decode steps after {profiler_wait_steps + profiler_warmup_steps} warmup (skipping prefill)")

        
        while n < gen_len:
            with self._maybe_record_function("decode_step"):
                logits = self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))
                next_token = sample_token(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
            
            n += 1
            generated_ids.append(next_token[:, -1].tolist())
            
            # Step profiler
            if enable_profiler:
                prof.step()
                # Stop profiler after wait + warmup + active steps
                if n == profiler_wait_steps + profiler_warmup_steps + profiler_active_steps:
                    self._profiling_enabled = False
                    prof.stop()
                    print(f"\nProfiler stopped after {profiler_active_steps} active decode steps")
                    print(f"Traces saved to: {profiler_output_dir}")
                    print(f"View with: tensorboard --logdir={profiler_output_dir}")
                    enable_profiler = False

        if benchmark == True:
            end = time.time()
            # print(f"\nPrefill {input_ids.size(1)} tokens | Generate {n} tokens in {round(end - start, 2)}s | Throughput: {round(self.batch_size * n / (end - start), 2)} tokens/s, Latency: {round((end - start)*1000 / n, 2)} ms/step | cached {self.kv_cache.get_kv_len()}\n")

        # feed new token to the model
        self.inference(input_ids=next_token, position_ids=self.get_ctx(next_token))

        # TODO(max410011): Uncomment these lines during memory usage evaluation
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        generated_ids = torch.LongTensor(generated_ids).t().tolist()

        if benchmark == True:
            return self.decode(generated_ids, skip_special_tokens=True), self.batch_size * n / (end - start)

        return self.decode(generated_ids, skip_special_tokens=True)