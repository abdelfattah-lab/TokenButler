/*
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
*/


#include <torch/extension.h>

void gather_copy(
    torch::Tensor values, torch::Tensor v_cache_buffer, torch::Tensor position_ids,
    int batch_size, int heads, int cpu_v_length, int gpu_v_length, int map_size);

void gather_copy_d2d_with_offsets(
    torch::Tensor keys,             // gpu keys
    torch::Tensor offsets,          // input, offsets computed from reorder_keys_and_compute_offsets, size as elements (numBlocks*256)
    torch::Tensor cnts,             // input, counts computed from reorder_keys_and_compute_offsets, size as numBlocks
    int batch_size, int heads, 
    int gpu_k_length, 
    int gpu_k_offset, 
    int gpu_k_stride, 
    int map_size);

void reorder_keys_and_compute_offsets(
    torch::Tensor cached_pos_ids, // inout, as cached previous position id as input, also reordered position ids, int64_t type
    torch::Tensor cur_pos_ids,    // input, incoming position id, int64_t type
    torch::Tensor offsets,        // output, offsets for gather_copy_with_offsets, size as numBlocks
    torch::Tensor cnts,           // output, counts to separate d2d and h2d, size as numBlocks
    int batch_size, int heads, int map_size);

void gather_copy_with_offsets(
    torch::Tensor values,           // input, cpu values
    torch::Tensor v_cache_buffer,   // inout, gpu values
    torch::Tensor temp,             // a temp gpu memory for copy, size same as single layer v_cache_buffer 
    torch::Tensor offsets,          // input, offsets computed from reorder_keys_and_compute_offsets, size as numBlocks, 
    torch::Tensor cnts,             // input, counts computed from reorder_keys_and_compute_offsets, size as numBlocks
    torch::Tensor signals,          // extra internal signals, all zeros sizes as numBlocks, size as numBlocks
    int batch_size, int heads, int cpu_v_length, int gpu_v_length, int gpu_v_offset, int gpu_v_stride, int map_size);

void apply_rotary_pos_emb(
    torch::Tensor x, torch::Tensor cos, torch::Tensor sin, torch::Tensor position_ids, torch::Tensor output,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos, int stride_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int half_dim);

void apply_rotary_pos_emb_new(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int half_dim);

void apply_rotary_pos_emb_new_v2(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int half_dim, int chunk_size);

void apply_rotary_pos_emb_push_cache(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    torch::Tensor cnts,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int stride_output_b, int stride_output_h, int stride_output_s,
    int offset_output_s_start, int offset_output_s_end,
    int half_dim, int chunk_size);

void apply_rotary_pos_emb_push_cache_opt(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    torch::Tensor cnts,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int stride_output_b, int stride_output_h, int stride_output_s,
    int offset_output_s_start, int offset_output_s_end,
    int half_dim, int chunk_size);

void apply_rotary_pos_emb_push_cache_opt_glm(
    torch::Tensor x, torch::Tensor cos_sin, torch::Tensor position_ids, torch::Tensor output,
    torch::Tensor cnts,
    int batch_size, int heads, int seq_len, int embed_dim,
    int stride_xb, int stride_xh, int stride_xs, int stride_xe,
    int stride_cos_sin,
    int stride_pid_b, int stride_pid_h, int stride_pid_s,
    int stride_output_b, int stride_output_h, int stride_output_s,
    int offset_output_s_start, int offset_output_s_end,
    int half_dim, int chunk_size);

void batch_gather_gemm(
    torch::Tensor a, torch::Tensor b,
    torch::Tensor cos, torch::Tensor sin,
    torch::Tensor position_ids,
    torch::Tensor output,
    int batch_size, int heads, int seq_len, int embed_dim, int rank, int sparse_budget,
    int max_seq_len, int chunk_size, torch::Tensor offset_array);

void batch_gemm_softmax(torch::Tensor A, torch::Tensor B,
                        torch::Tensor D, torch::Tensor Norm, torch::Tensor Sum,
                        torch::Tensor Softmax, int batch_count, int m, int n,
                        int k, float alpha, float beta);