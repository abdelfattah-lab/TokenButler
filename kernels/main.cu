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
#include "functions.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_copy", &gather_copy, "Gather-Copy (CUDA)");
    m.def("gather_copy_d2d_with_offsets", &gather_copy_d2d_with_offsets, "Gather-Copy D2D with offsets for keys (CUDA)");
    m.def("reorder_keys_and_compute_offsets", &reorder_keys_and_compute_offsets, "Reorder keys and compute offsets (CUDA)");
    m.def("gather_copy_with_offsets", &gather_copy_with_offsets, "Gather-Copy with offsets (CUDA)");
    m.def("apply_rotary_pos_emb", &apply_rotary_pos_emb, "Apply rotary position embedding (CUDA)");
    m.def("apply_rotary_pos_emb_new", &apply_rotary_pos_emb_new, "Apply rotary positional embeddings (CUDA)");
    m.def("apply_rotary_pos_emb_new_v2", &apply_rotary_pos_emb_new_v2, "Apply rotary positional embeddings version 2 (CUDA)");
    m.def("apply_rotary_pos_emb_push_cache", &apply_rotary_pos_emb_push_cache, "Apply rotary positional embeddings with cache pushing (CUDA)");
    m.def("apply_rotary_pos_emb_push_cache_opt", &apply_rotary_pos_emb_push_cache_opt, "Apply rotary positional embeddings with cache pushing (CUDA)");
    m.def("apply_rotary_pos_emb_push_cache_opt_glm", &apply_rotary_pos_emb_push_cache_opt_glm, "Apply rotary positional embeddings with cache pushing (CUDA) for GLM");
    m.def("batch_gather_gemm", &batch_gather_gemm, "Batch Gather GEMM (CUDA)");
    m.def("batch_gemm_softmax", &batch_gemm_softmax, "Batch GEMM Softmax (CUDA)");
}