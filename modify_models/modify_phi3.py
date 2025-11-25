import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import gc

import traceback
import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.phi3.modeling_phi3 import (
    apply_rotary_pos_emb,
    Phi3Config,
    Phi3Attention,
    Phi3RotaryEmbedding,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from utils import repeat_kv, sorted_index_to_variable_mask, SlidingWindowCache, enforce_sliding_window, threshold_to_mask
from utils import calculate_hit_metrics
from transformers.cache_utils import DynamicCache
from predictor import TokenImportancePredictorAttentive, PredictorDynamicCache, HeadImportancePredictor, attention_mse_loss, attention


from triton_kernels.flash_attn import attention
from triton_kernels.flash_attn_mse_loss import attention_mse_loss

class Phi3LongRoPEScaledRotaryEmbedding(Phi3RotaryEmbedding):
    def __init__(self, config, dim, device=None):
        # HF Phi3RotaryEmbedding expects `config` as first arg in your version
        super().__init__(config, device=device)

        # Now override / add attributes for LongRoPE behavior
        self.dim = dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta

        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=x.device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=x.device)

        inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)

            scale = self.max_position_embeddings / self.original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

            cos = emb.cos() * scaling_factor
            sin = emb.sin() * scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Phi3AttentionExperimental(nn.Module):
    def __init__(
        self,
        config: Phi3Config,
        producer: Optional["Phi3AttentionExperimental"] = None,
        layer_idx: int = 0,
        producer_frequency: int = 1,
        is_predictor_owner: bool = False,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.inference_mode = False
        self.layer_idx = layer_idx
        self.producer_frequency = max(1, int(producer_frequency))

        # Module that owns the predictor outputs for this layer group.
        self.producer: Optional["Phi3AttentionExperimental"] = producer

        self.token_sparse_method = None
        self.sparse_aggression = None
        self.stream_llm_start_size = None
        self.dDash = None
        self.intdim = None
        self.attn_reduce_factor = None
        self.head_attn_reduce_factor = None
        self.effective_sparsity = None
        self.min_sparse_index = None
        self.pred_hid_size = self.hidden_size
        # Each predictor call produces one query tensor per consumer layer in the group.
        self.num_layers_pred = self.producer_frequency
        self.num_tok_per_page = None
        self.calc_hitrates = False
        self.flash_attn = False
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.rope_scaling = config.rope_scaling
        self.train_headpredictor = False
        self.calibrate_thresholds = False
        self.test_with_thresholds = False
        self.late_context_upweight = False
        self.softmax_causal_loss_mse = False
        self.softmax_causal_loss_ce = False
        self.pairwise_loss = False
        self.pairwise_topk_ratio = 0.1  # fraction of keys used as pos/neg in pairwise loss
        self.old_predictor = None
        self.mode = "balanced"
        self.tokenbutler_variant = "tokenbutler_project"
        self.lookahead = 0

        # Whether this layer runs the predictor (i.e. group root).
        self.is_predictor_owner = is_predictor_owner
        # Decode-time gating knobs.
        self.target_sparsity = None
        self.target_keep_tokens = None
        self._dense_kv_cutoff = 0
        self.always_dense_decode_tokens = 1

        self.low_recall_first = {}

        if self.config._name_or_path not in self.low_recall_first:
            self.lowrecall_tuples = []
        else:
            self.lowrecall_tuples = self.low_recall_first[self.config._name_or_path]

        if self.layer_idx > 0:
            self.mseloss = MSELoss(reduction='none')
            self.msemagn_loss = None
            self.headmseloss = MSELoss(reduction='none')
            self.headmsemagn_loss = None

        if self.is_predictor_owner:
            self.q_importance = None  # Shared mask across layers during inference
            self.k_importance = None
            self.head_importances = None
            self.actmagn_masklist = {}
            self.available_tokens = {}

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.qkv_proj = nn.Linear(self.hidden_size, op_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.rope_scaling is None:
            # Your HF Phi3RotaryEmbedding takes config as first arg, so do the same:
            self.rotary_emb = Phi3RotaryEmbedding(self.config)
        else:
            scaling_type = self.config.rope_scaling["type"]
            if scaling_type == "longrope":
                # ✅ config first, then dim
                self.rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(self.config, self.head_dim)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _compute_global_head_keep(self):
        """
        Build a tensor  shape [num_hidden_layers, num_heads] that contains
        the fraction of keys to keep for *every* (layer, head) pair,
        scaled so the **model-wide** average equals self.sparse_aggression.
        Called once, cached in self._global_head_keep.
        """
        L, H = self.num_hidden_layers, self.num_heads
        keep = torch.full((L, H), float(self.sparse_aggression))

        bad_pairs = self.lowrecall_tuples
        if not bad_pairs:
            self._global_head_keep = keep
            return

        keep_max, keep_min = 1.0, float(self.sparse_aggression)
        N = len(bad_pairs)
        for rank, (head_idx, layer_idx) in enumerate(bad_pairs):
            frac = rank / (N - 1 + 1e-5)
            keep[layer_idx, head_idx] = keep_max - frac * (keep_max - keep_min)

        total_heads = L * H
        scale = (self.sparse_aggression * total_heads) / keep.sum()
        keep *= scale
        keep.clamp_(max=1.0)

        self._global_head_keep = keep

    def build_head_keep_ratios(self):
        """
        Return the [num_heads] vector for *this* layer, using
        model-global calibration. Safe to call every forward(); the
        table is computed once and cached.
        """
        if not hasattr(self, "_global_head_keep"):
            self._compute_global_head_keep()
        return self._global_head_keep[self.layer_idx].to(next(self.parameters()).device)

    def update_predictor(self):
        attn_device = next(self.parameters()).device
        self.sparse_token_predictor = TokenImportancePredictorAttentive(
            self.config,
            self.pred_hid_size,
            self.num_heads,
            self.num_layers_pred,
            dropout=0.1,
            dDash=self.dDash,
            intdim=self.intdim,
            attn_reduce_factor=self.attn_reduce_factor,
            predictor_variant=getattr(self, "tokenbutler_variant", "tokenbutler_project"),
        ).to(attn_device)
        self.sparse_token_predictor.flash_attn = self.flash_attn
        if self.train_headpredictor:
            self.sparse_head_predictor = HeadImportancePredictor(
                self.config,
                self.pred_hid_size,
                self.num_heads,
                self.num_layers_pred,
                dropout=0.1,
                dDash=self.dDash,
                intdim=self.intdim,
                attn_reduce_factor=self.head_attn_reduce_factor,
            ).to(attn_device)
            self.sparse_head_predictor.flash_attn = self.flash_attn

    def set_token_sparsity(self):
        assert self.token_sparse_method is not None, "Set token sparse method first!"
        method = self.token_sparse_method

        if method is not None:
            try:
                mname = self.config._name_or_path.split("/")[-1]
                read_path = f"threshold_calibs/{mname}/{method}.pkl"
                threshold_model_dictionary = torch.load(read_path)
                self.tok_calibration_set = threshold_model_dictionary
            except Exception:
                pass

        self.target_sparsity = None
        self.target_keep_tokens = None
        self.sparse_aggression = None
        self.head_keep = None

        if method.startswith("fixed_"):
            spec = method.split("_", 1)[1]

            if spec.endswith("pc"):
                if self.layer_idx == 0:
                    self.target_sparsity = 0.0
                    self.sparse_aggression = 1.0
                else:
                    x = float(spec[:-2])
                    self.target_sparsity = x / 100.0
                    self.sparse_aggression = 1.0 - self.target_sparsity
                self.head_keep = self.build_head_keep_ratios()
            elif spec.endswith("tok"):
                y = int(spec[:-3])
                if self.layer_idx == 0:
                    self.target_keep_tokens = None
                    self.sparse_aggression = 1.0
                else:
                    self.target_keep_tokens = max(1, y)
                    nominal_L = getattr(self, "nominal_seq_len", None)
                    if nominal_L is None:
                        nominal_L = getattr(self.config, "max_position_embeddings", self.max_position_embeddings)
                    head = getattr(self, "min_sparse_index", 0) or 0
                    tail = getattr(self, "sliding_window", 0) or 0
                    eff_L = max(1, nominal_L - head - tail)
                    keep_frac = min(1.0, float(self.target_keep_tokens) / eff_L)
                    self.sparse_aggression = keep_frac
                    self.head_keep = None
            else:
                raise ValueError(f"Unknown fixed sparsity spec '{spec}' in token_sparse_method='{method}'")
        else:
            raise ValueError(
                f"Unsupported token sparsity method '{method}'. "
                "Use 'fixed_xpc' (e.g. fixed_65pc) or 'fixed_ytok' (e.g. fixed_128tok)."
            )
            

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _should_apply_sparse_decode(self, q_len: int, kv_seq_len: int) -> bool:
        if q_len != 1:
            return False
        if self._dense_kv_cutoff == 0:
            self._dense_kv_cutoff = kv_seq_len
            return False
        return kv_seq_len > self._dense_kv_cutoff

    def _build_decode_mask_fixed(
        self,
        importance_scores: torch.Tensor,
        attention_mask: torch.Tensor,
        min_sparse_index: Optional[int],
    ) -> torch.Tensor:
        bsz, num_heads, q_len, key_len = importance_scores.shape
        assert q_len == 1, "Decode-time mask only valid for q_len == 1"
        device = importance_scores.device
        dtype = importance_scores.dtype

        attn_valid = attention_mask[:, :, -1:, :] == 0
        candidate_mask = attn_valid.expand(bsz, num_heads, 1, key_len)

        if min_sparse_index is not None and min_sparse_index > 0:
            clamp_idx = min(min_sparse_index, key_len)
            candidate_mask[..., :clamp_idx] = False

        if self.sliding_window is not None and self.sliding_window > 0:
            win = min(self.sliding_window, key_len)
            candidate_mask[..., -win:] = False

        if not candidate_mask.any():
            return torch.zeros_like(importance_scores, dtype=dtype, device=device)

        method = self.token_sparse_method or ""
        candidate_counts = candidate_mask.sum(dim=-1, keepdim=True)

        if "pc" in method:
            if self.sparse_aggression is None:
                raise ValueError("sparse_aggression must be set for fixed_xpc")
            if getattr(self, "head_keep", None) is not None:
                head_keep = self.head_keep.to(device)
            else:
                head_keep = torch.full((self.num_heads,), float(self.sparse_aggression), device=device)
            head_keep = head_keep.clamp(min=0.0, max=1.0).view(1, self.num_heads, 1, 1)
            keep_counts = (head_keep * candidate_counts.float()).floor()
            keep_counts = keep_counts.clamp(min=1)
            keep_counts = torch.minimum(keep_counts, candidate_counts).long()
        elif "tok" in method:
            if self.target_keep_tokens is None or self.target_keep_tokens <= 0:
                return torch.zeros_like(importance_scores, dtype=dtype, device=device)
            keep_counts = torch.minimum(
                candidate_counts,
                torch.full_like(candidate_counts, self.target_keep_tokens, dtype=torch.long),
            ).clamp(min=1)
        else:
            raise ValueError(f"token_sparse_method '{method}' is not a fixed_* scheme")

        scores = importance_scores.clone().masked_fill(~candidate_mask, float("-inf"))
        _, sorted_idx = scores.sort(dim=-1, descending=True)
        B, H, _, K = sorted_idx.shape
        rank = torch.empty_like(sorted_idx, dtype=torch.long)
        arange_K = torch.arange(K, device=sorted_idx.device, dtype=torch.long).view(1, 1, 1, K).expand_as(sorted_idx)
        rank.scatter_(-1, sorted_idx, arange_K)
        keep_mask = (~candidate_mask) | (rank < keep_counts)

        mask_tensor = torch.zeros_like(importance_scores, dtype=dtype, device=device)
        mask_tensor = mask_tensor.masked_fill(~keep_mask, float("-inf"))

        if min_sparse_index is not None and min_sparse_index > 0:
            clamp_idx = min(min_sparse_index, key_len)
            mask_tensor[..., :clamp_idx] = 0.0
        if self.sliding_window is not None and self.sliding_window > 0:
            win = min(self.sliding_window, key_len)
            mask_tensor[..., -win:] = 0.0

        return mask_tensor

    def _get_group_slot_index(self) -> Optional[int]:
        if self.layer_idx == 0 or self.producer is None:
            return None
        return (self.layer_idx - 1) % self.producer_frequency

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Union[DynamicCache, PredictorDynamicCache]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()
        Ltrack = hidden_states.size(1)

        if q_len != 1:  # this is prefill stage for first token output, reset q-k importance tensors
            self.q_importance = None
            self.k_importance = None
            self.head_importances = None
            
        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        evalmode = self.eval_llm_mode
        num_tokens_to_keep = int(q_len * self.sparse_aggression)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # --- 2.2: RoPE + cache semantics (match stock Phi-3, incl. LongRoPE & sliding window) ---
        kv_seq_len = key_states.shape[-2]
        past_kv_len = 0
        if past_key_value is not None:
            # DynamicCache exposes `get_usable_length`, custom caches might only expose `get_seq_length`.
            if hasattr(past_key_value, "get_usable_length"):
                past_kv_len = past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            elif hasattr(past_key_value, "get_seq_length"):
                past_kv_len = past_key_value.get_seq_length(self.layer_idx)
            kv_seq_len += past_kv_len

        # For Phi-3-mini-128k-instruct this will be a LongRoPE embedding under the hood.
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if use_cache and past_key_value is not None:
            # Pass RoPE cos/sin into the cache (same as stock Phi-3 behavior).
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        if self.inference_mode and use_cache and q_len > 1 and kv_seq_len == q_len:
            self._dense_kv_cutoff = kv_seq_len + self.always_dense_decode_tokens

        final_mask = None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_len = key_states.size(2)
        bsz, q_len = query_states.size(0), query_states.size(2)

        # --- 2.3: attention mask when `attention_mask is None` ---
        # If we weren't given a 4D mask (e.g. calling this module directly),
        # build the same mask HF would via `_prepare_4d_causal_attention_mask`.
        if attention_mask is None:
            sliding_window = getattr(self, "sliding_window", None)
            if sliding_window is None:
                sliding_window = getattr(self.config, "sliding_window", None)

            attention_mask = _prepare_4d_causal_attention_mask(
                None,                    # no 2D pad mask provided
                (bsz, q_len),            # (batch_size, query_length)
                hidden_states,           # only dtype/device are used
                past_kv_len,             # length of KV cache *before* adding current tokens
                sliding_window=sliding_window,
            )

        if self.inference_mode:
            min_sparse_index = self.min_sparse_index
            with torch.no_grad():
                if evalmode == "ExpPred":
                    if self.layer_idx > 0:
                        slot_idx = self._get_group_slot_index()
                        q_importance_tensor = self.producer.q_importance[:, slot_idx, :, :].float().to(query_states.device) # [BH, Lq, D']
                        k_importance_tensor = self.producer.k_importance[:, slot_idx, :, :].float().to(key_states.device) # [BH, Lk, D']
                        importance_mask = torch.bmm(q_importance_tensor, k_importance_tensor.transpose(-2, -1)) / math.sqrt(self.dDash) # [BH, Lq, Lk]
                        importance_mask = importance_mask.view(bsz, self.num_heads, q_len, key_len) # [B, H, Lq, Lk]
                        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                        if self.calc_hitrates:
                            self.tok_hit_acc, self.tok_mean_rank_corr, self.tok_max_rank_corr = calculate_hit_metrics(
                                estimated_importance=nn.functional.softmax(importance_mask + attention_mask, dim=-1),
                                true_importance=nn.functional.softmax(attn_weights + attention_mask, dim=-1),
                                top_k_ratio=0.5
                            )
                        if self.calibrate_thresholds:
                            ### Threshold variance investigation
                            unadj_importance_mask = importance_mask.clone()
                            importance_mask = torch.softmax(importance_mask + attention_mask, dim=-1)
                            sorted_indices = torch.argsort(importance_mask, dim=-1, descending=True)
                            sorted_indices = sorted_indices[:, :, -q_len:, :]
                            sorted_values, sorted_ix = torch.sort(importance_mask, dim=-1)
                            sorted_true_values, _ = torch.sort(torch.gather(unadj_importance_mask, dim=-1, index=sorted_ix), dim=-1)
                            true_thresholds = sorted_true_values[:, :, :, int(importance_mask.size(-1) * self.sparse_aggression)]
                            thresholds = sorted_values[:, :, :, int(importance_mask.size(-1) * self.sparse_aggression)]
                            self.true_threshmean = true_thresholds
                            self.threshmean = thresholds
                        if self.test_with_thresholds:
                            unadj_importance_mask = importance_mask.clone()
                            perhead_thresholds = self.tok_calibration_set[self.layer_idx - 1].to(unadj_importance_mask.device) # 0 does not have calibration data.
                            mask_tensor = threshold_to_mask(unadj_importance_mask, perhead_thresholds, min_sparse_index, bsz, q_len, key_len)
                        else:
                            importance_probs = torch.softmax(importance_mask + attention_mask, dim=-1)
                            apply_sparse_decode = self._should_apply_sparse_decode(q_len, key_len)
                            if (
                                apply_sparse_decode
                                and self.token_sparse_method is not None
                                and self.token_sparse_method.startswith("fixed_")
                            ):
                                mask_tensor = self._build_decode_mask_fixed(
                                    importance_probs,
                                    attention_mask,
                                    min_sparse_index,
                                )
                            else:
                                _, sorted_indices = importance_probs.sort(dim=-1, descending=True)  # [B, H, q_len, key_len]
                                sorted_indices = sorted_indices[:, :, -q_len:, :]
                                if q_len == 1:
                                    mask_tensor = torch.zeros_like(importance_probs)
                                    sorted_indices = sorted_indices[:, :, :, int(self.sparse_aggression * key_len):]
                                    mask_tensor.scatter_(-1, sorted_indices, float('-inf'))
                                    mask_tensor[:, :, :, :min_sparse_index] = 0.0
                                    if self.sliding_window is not None:
                                        mask_tensor[:, :, :, -self.sliding_window:] = 0.0
                                else:
                                    mask_tensor = sorted_index_to_variable_mask(
                                        sorted_indices,
                                        attention_mask,
                                        min_sparse_index,
                                        bsz,
                                        q_len,
                                        key_len,
                                        self.head_keep.to(sorted_indices.device),
                                        sliding_window=self.sliding_window
                                    )
                        if self.sliding_window is not None:
                            if not hasattr(self, "window_cache"):
                                self.window_cache = SlidingWindowCache(
                                    max_seq_len=1024,
                                    sliding_window=self.sliding_window,
                                    device=mask_tensor.device,
                                )
                            window = self.window_cache.get_window(q_len, key_len)
                            mask_tensor = enforce_sliding_window(mask_tensor, window)
                        final_mask = mask_tensor

                        self.final_mask_investigate = final_mask
                        attn_weights = attn_weights + attention_mask
                        # if q_len == 1:
                        attn_weights = attn_weights + mask_tensor
                    else:
                        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                        attn_weights = attn_weights + attention_mask
                else:
                    raise ValueError(f"Unknown eval mode {evalmode}")
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        else:
            if self.flash_attn:
                if self.layer_idx > 0:
                    # Token hit-rates cannot be calculated if using flash attention.
                    self.tok_hit_acc = 0
                    q_importance_tensor = self.producer.q_importance[:, self.layer_idx % self.producer_frequency, :, :].float().to(query_states.device) # [BH, Lq, D']
                    k_importance_tensor = self.producer.k_importance[:, self.layer_idx % self.producer_frequency, :, :].float().to(key_states.device) # [BH, Lk, D']
                    q_importance_tensor = q_importance_tensor.view(bsz, self.num_heads, q_len, self.dDash)
                    k_importance_tensor = k_importance_tensor.view(bsz, self.num_heads, key_len, self.dDash)
                    device_index = query_states.device.index
                    assert self.lookahead == 0, "Lookahead not supported with flash attention yet. Please disable --flash_attn"
                    with torch.cuda.device(device_index):
                        attn_output, mse_loss = attention_mse_loss(query_states.contiguous().to(torch.float16),
                                                                    key_states.contiguous().to(torch.float16),
                                                                    value_states.contiguous().to(torch.float16),
                                                                    q_importance_tensor.contiguous().to(torch.float16),
                                                                    k_importance_tensor.contiguous().to(torch.float16), 
                                                                    True
                                                                    )
                    self.tok_hit_acc, self.tok_mean_rank_corr, self.tok_max_rank_corr = 0, 0, 0
                    attn_output = attn_output.to(query_states.dtype)
                    if not torch.isnan(mse_loss):
                        self.msemagn_loss = mse_loss
                    else:
                        raise ValueError(f"NaN loss detected: {mse_loss}")
                else:
                    attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=None, is_causal=True)
            else:
                # Teacher attention logits (no mask yet)
                attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

                if self.layer_idx > 0:
                    # Predictor logits (importance mask)
                    q_importance_tensor = self.producer.q_importance[:, self.layer_idx % self.producer_frequency, :, :].float().to(query_states.device)  # [BH, Lq, D']
                    k_importance_tensor = self.producer.k_importance[:, self.layer_idx % self.producer_frequency, :, :].float().to(key_states.device)    # [BH, Lk, D']
                    importance_mask = torch.bmm(q_importance_tensor, k_importance_tensor.transpose(-2, -1)) / math.sqrt(self.dDash)                      # [BH, Lq, Lk]
                    importance_mask = importance_mask.view(bsz, self.num_heads, q_len, key_len)                                                          # [B, H, Lq, Lk]

                    # ---- build teacher & student logits used for auxiliary loss ----
                    if self.lookahead == 0:
                        # standard causal training: apply mask before softmax-based losses
                        if attention_mask is not None:
                            teacher_logits = attn_weights + attention_mask      # [B,H,Lq,Lk]
                            student_logits = importance_mask + attention_mask   # [B,H,Lq,Lk]
                        else:
                            teacher_logits = attn_weights
                            student_logits = importance_mask
                    else:
                        # lookahead training: align row t (student) with row t+lookahead (teacher)
                        # we mirror your previous raw-logit MSE semantics (no mask here)
                        teacher_logits = attn_weights[:, :, self.lookahead:, :]        # [B,H,Lq_eff,Lk]
                        student_logits = importance_mask[:, :, :-self.lookahead, :]    # [B,H,Lq_eff,Lk]

                    # ---- Loss selection: exactly one of these should be True ----
                    if self.softmax_causal_loss_mse:
                        # 1) MSE between teacher and predictor distributions
                        target_dist = F.softmax(teacher_logits, dim=-1)
                        pred_dist   = F.softmax(student_logits, dim=-1)
                        loss = self.mseloss(pred_dist, target_dist)                    # [B,H,Lq_eff,Lk]
                        self.msemagn_loss = 1024*loss.mean(dim=(-1, -2)).mean()             # scalar

                    elif self.softmax_causal_loss_ce:
                        # 2) Cross-entropy / KL-like loss between distributions
                        target_dist = F.softmax(teacher_logits, dim=-1).detach()
                        pred_dist   = F.softmax(student_logits, dim=-1)
                        ce = -(target_dist * (pred_dist + 1e-9).log()).sum(dim=-1)     # [B,H,Lq_eff]
                        self.msemagn_loss = 0.1 * ce.mean()                                  # scalar

                    elif getattr(self, "pairwise_loss", False):
                        # 3) Pairwise ranking loss (logistic) on teacher-defined pairs.
                        #
                        # We want teacher logits WITH the causal mask applied when forming pairs,
                        # so we don't accidentally pick future / invalid tokens.
                        if attention_mask is not None:
                            full_teacher_logits = attn_weights + attention_mask
                            full_student_logits = importance_mask + attention_mask
                        else:
                            full_teacher_logits = attn_weights
                            full_student_logits = importance_mask

                        if self.lookahead == 0:
                            teacher_logits_pw = full_teacher_logits            # [B,H,Lq,Lk]
                            student_logits_pw = full_student_logits            # [B,H,Lq,Lk]
                            attn_mask_eff   = attention_mask                   # [1,1,Lq,Lk] or None
                        else:
                            # Align row t (student) with row t+lookahead (teacher)
                            teacher_logits_pw = full_teacher_logits[:, :, self.lookahead:, :]      # [B,H,Lq_eff,Lk]
                            student_logits_pw = full_student_logits[:, :, :-self.lookahead, :]     # [B,H,Lq_eff,Lk]
                            attn_mask_eff = None if attention_mask is None else attention_mask[:, :, self.lookahead:, :]

                        # Teacher probabilities
                        teacher_probs = F.softmax(teacher_logits_pw, dim=-1)    # [B,H,Lq_eff,Lk]
                        B_eff, H_eff, Lq_eff, Lk_eff = teacher_probs.shape

                        # Valid (non-masked) positions: attention_mask == 0
                        if attn_mask_eff is not None:
                            # attn_mask_eff: [1,1,Lq_eff,Lk]  -> broadcast to [B,H,Lq_eff,Lk]
                            valid_mask = (attn_mask_eff == 0).expand(B_eff, H_eff, Lq_eff, Lk_eff)
                        else:
                            valid_mask = torch.ones_like(teacher_probs, dtype=torch.bool)

                        # If some rows have no valid keys, we should ignore them in the loss
                        valid_counts = valid_mask.reshape(-1, Lk_eff).sum(-1)      # [B_eff*H_eff*Lq_eff]
                        has_valid = (valid_counts > 0).reshape(B_eff, H_eff, Lq_eff)  # [B_eff,H_eff,Lq_eff]

                        # Hyperparameter: fraction of keys to use for pos/neg sampling
                        topk_ratio = getattr(self, "pairwise_topk_ratio", 0.2)
                        K = max(1, int(topk_ratio * Lk_eff))

                        # --- choose positives: highest-prob valid tokens ---
                        # invalid positions get prob -inf so they never appear in topk
                        probs_for_top = teacher_probs.masked_fill(~valid_mask, float("-inf"))
                        top_vals, top_idx = probs_for_top.topk(K, dim=-1)       # [B,H,Lq_eff,K]

                        # --- choose negatives: lowest-prob valid tokens ---
                        # invalid positions get prob +1.0 so they never appear among the lowest
                        probs_for_bot = teacher_probs.masked_fill(~valid_mask, 1.0)
                        bot_vals, bot_idx = probs_for_bot.topk(K, dim=-1, largest=False)  # [B,H,Lq_eff,K]

                        # Clamp student logits to avoid inf / NaN in margin
                        student_logits_pw = student_logits_pw.clamp(min=-1e4, max=1e4)

                        # Gather student scores at those positions
                        s = student_logits_pw                                      # [B,H,Lq_eff,Lk]
                        s_pos = s.gather(-1, top_idx)                              # [B,H,Lq_eff,K]
                        s_neg = s.gather(-1, bot_idx)                              # [B,H,Lq_eff,K]

                        margin = s_pos - s_neg                                     # [B,H,Lq_eff,K]
                        pairwise = F.softplus(-margin)                             # log(1 + exp(-margin))

                        # Zero out rows with no valid keys (to avoid NaNs from degenerate rows)
                        if attn_mask_eff is not None:
                            pairwise = pairwise * has_valid.unsqueeze(-1).to(pairwise.dtype)

                        self.msemagn_loss = pairwise.mean()

                    elif getattr(self, "pairwise_ce_loss", False):
                        # 3b) Pairwise CE loss: match 2-way distributions over (pos, neg) tokens

                        # Use masked logits so we never pick invalid/future tokens
                        if attention_mask is not None:
                            full_teacher_logits = attn_weights + attention_mask
                            full_student_logits = importance_mask + attention_mask
                        else:
                            full_teacher_logits = attn_weights
                            full_student_logits = importance_mask

                        if self.lookahead == 0:
                            teacher_logits_pw = full_teacher_logits            # [B,H,Lq,Lk]
                            student_logits_pw = full_student_logits            # [B,H,Lq,Lk]
                            attn_mask_eff   = attention_mask                   # [1,1,Lq,Lk] or None
                        else:
                            teacher_logits_pw = full_teacher_logits[:, :, self.lookahead:, :]      # [B,H,Lq_eff,Lk]
                            student_logits_pw = full_student_logits[:, :, :-self.lookahead, :]     # [B,H,Lq_eff,Lk]
                            attn_mask_eff = None if attention_mask is None else attention_mask[:, :, self.lookahead:, :]

                        # Teacher & predictor probabilities
                        teacher_probs = F.softmax(teacher_logits_pw, dim=-1)    # [B,H,Lq_eff,Lk]
                        pred_probs    = F.softmax(student_logits_pw, dim=-1)    # [B,H,Lq_eff,Lk]
                        B_eff, H_eff, Lq_eff, Lk_eff = teacher_probs.shape

                        # Valid (non-masked) positions
                        if attn_mask_eff is not None:
                            valid_mask = (attn_mask_eff == 0).expand(B_eff, H_eff, Lq_eff, Lk_eff)
                        else:
                            valid_mask = torch.ones_like(teacher_probs, dtype=torch.bool)

                        valid_counts = valid_mask.reshape(-1, Lk_eff).sum(-1)
                        has_valid = (valid_counts > 0).reshape(B_eff, H_eff, Lq_eff)

                        # Fraction of keys to use for pos/neg sampling
                        topk_ratio = getattr(self, "pairwise_topk_ratio", 0.2)
                        K = max(1, int(topk_ratio * Lk_eff))

                        # Positives: highest-prob valid tokens
                        probs_for_top = teacher_probs.masked_fill(~valid_mask, float("-inf"))
                        _, top_idx = probs_for_top.topk(K, dim=-1)                     # [B,H,Lq_eff,K]

                        # Negatives: lowest-prob valid tokens
                        probs_for_bot = teacher_probs.masked_fill(~valid_mask, 1.0)
                        _, bot_idx = probs_for_bot.topk(K, dim=-1, largest=False)      # [B,H,Lq_eff,K]

                        # Gather teacher + predictor probs for pos/neg
                        t_pos = teacher_probs.gather(-1, top_idx)                      # [B,H,Lq_eff,K]
                        t_neg = teacher_probs.gather(-1, bot_idx)
                        p_pos = pred_probs.gather(-1, top_idx)
                        p_neg = pred_probs.gather(-1, bot_idx)

                        # Build 2-way distributions per pair: [ ..., K, 2 ]
                        t_pair = torch.stack([t_pos, t_neg], dim=-1)
                        p_pair = torch.stack([p_pos, p_neg], dim=-1)

                        # Normalize along the 2-way dimension
                        t_pair = t_pair / (t_pair.sum(dim=-1, keepdim=True) + 1e-9)
                        p_pair = p_pair / (p_pair.sum(dim=-1, keepdim=True) + 1e-9)

                        # CE over the 2-way pair distribution
                        pair_ce = -(t_pair * (p_pair + 1e-9).log()).sum(dim=-1)       # [B,H,Lq_eff,K]

                        # Zero out rows with no valid keys
                        if attn_mask_eff is not None:
                            pair_ce = pair_ce * has_valid.unsqueeze(-1).to(pair_ce.dtype)

                        self.msemagn_loss = pair_ce.mean()
                    else:
                        raise ValueError("No loss selected for token importance predictor!")

                    # ---- Metrics (optional) ----
                    if self.calc_hitrates:
                        est   = F.softmax(student_logits, dim=-1)
                        truth = F.softmax(teacher_logits, dim=-1)
                        self.tok_hit_acc, self.tok_mean_rank_corr, self.tok_max_rank_corr = calculate_hit_metrics(
                            estimated_importance=est,
                            true_importance=truth,
                            top_k_ratio=0.5
                        )
                # Main model attention path: always use teacher logits + causal mask
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)

        if self.layer_idx > 0 and self.train_headpredictor:
            slot_idx = self._get_group_slot_index()
            head_importance_tensor = self.producer.head_importances[:, :, :, slot_idx].float().to(attn_output.device)
            attn_head_weights = attn_output.mean(dim=-1).permute(0, 2, 1)
            self.headmsemagn_loss = self.headmseloss(attn_head_weights, head_importance_tensor).mean()

            if self.calc_hitrates:
                self.head_hit_acc, self.head_mean_rank_corr, self.head_max_rank_corr = calculate_hit_metrics(
                    estimated_importance=head_importance_tensor,
                    true_importance=attn_head_weights,
                    top_k_ratio=0.5
                )
        else:
            self.headmsemagn_loss = 0
            if self.calc_hitrates:
                self.head_hit_acc, self.head_mean_rank_corr, self.head_max_rank_corr = 0, 0, 0

            
        checkeverytime = hasattr(self, 'test_with_thresholds')
        if checkeverytime:
            checkeverytime = self.test_with_thresholds
        if final_mask is not None and q_len == 1:
            if self.effective_sparsity is None or checkeverytime:
                true_mask = final_mask + attention_mask  # {0, -inf}

                candidate_mask = (~attention_mask.bool())
                min_sparse_index = getattr(self, "min_sparse_index", None)
                if min_sparse_index is not None and min_sparse_index > 0:
                    clamp_idx = min(min_sparse_index, true_mask.size(-1))
                    candidate_mask[..., :clamp_idx] = False
                if self.sliding_window is not None and self.sliding_window > 0:
                    win = min(self.sliding_window, true_mask.size(-1))
                    candidate_mask[..., -win:] = False

                if candidate_mask.any():
                    total_deact = (true_mask.bool() & candidate_mask).sum(dim=-1)
                    causal_deact = (attention_mask.bool() & candidate_mask).sum(dim=-1)
                    additional_deact = (total_deact - causal_deact)
                    num_candidates = candidate_mask.sum(dim=-1)
                    effective_sparsity = 100 * (additional_deact.float() / num_candidates.float()).mean().item()
                else:
                    effective_sparsity = 0.0
                self.effective_sparsity = effective_sparsity
                print("Effective Sparsity:", effective_sparsity, "%\t Sequence Length:", q_len)

        if self.layer_idx == 0:
            if self.effective_sparsity is None:
                self.effective_sparsity = 0.0

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if self.is_predictor_owner:
            try:
                q_importance, k_importance = self.sparse_token_predictor(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,  # the same single cache
                    use_cache=use_cache,
                    layer_idx=self.layer_idx,       # or pass 0
                )
                if self.train_headpredictor:
                    head_importances, past_key_value_hp = self.sparse_head_predictor(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value_hp,
                        use_cache=use_cache
                    )
                    head_importances = head_importances.view(bsz, q_len, self.num_heads, self.num_hidden_layers) # [B L H N]
                q_len = attn_output.size(1)
                k_len = k_importance.size(-1)
            except:
                print(traceback.format_exc())
                import pdb; pdb.set_trace()

            self.q_importance = q_importance
            self.k_importance = k_importance

            if self.train_headpredictor:
                if self.head_importances is None:
                    self.head_importances = head_importances
                else:
                    self.head_importances = torch.cat([self.head_importances, head_importances], dim=1)

        if not output_attentions:
            attn_weights = None

        # Match the original Phi3Attention interface:
        # (attn_output, attn_weights, present_key_value)
        # present_key_value is just `past_key_value` (the Cache object) possibly updated in-place.
        return attn_output, attn_weights, past_key_value
        # if not output_attentions:
        #     attn_weights = None

        # return attn_output, attn_weights

def convert_kvcache_experimental(model, config, producer_frequency: int):
    layer_idx_counter = 0
    group_roots: dict[int, Phi3AttentionExperimental] = {}

    def recurse(parent_module: nn.Module, prefix: str = ""):
        nonlocal layer_idx_counter

        for name, child in list(parent_module._modules.items()):
            full_name = f"{prefix}.{name}" if prefix else name

            if len(list(child.children())) > 0:
                recurse(child, full_name)
            if child.__class__.__name__.endswith("Phi3Attention"):
                try:
                    ref_param = next(child.parameters())
                    target_device = ref_param.device
                    orig_dtype = ref_param.dtype
                except StopIteration:
                    target_device = torch.device("cpu")
                    orig_dtype = torch.float32

                layer_idx = layer_idx_counter
                is_owner = (layer_idx % producer_frequency == 0)

                if layer_idx == 0:
                    producer = None
                else:
                    base_idx = ((layer_idx - 1) // producer_frequency) * producer_frequency
                    producer = group_roots[base_idx]

                new_attn = Phi3AttentionExperimental(
                    config=config,
                    producer=producer,
                    layer_idx=layer_idx,
                    producer_frequency=producer_frequency,
                    is_predictor_owner=is_owner,
                ).to(device=target_device, dtype=orig_dtype)

                new_attn.load_state_dict(child.state_dict(), strict=False)

                if is_owner:
                    group_roots[layer_idx] = new_attn

                parent_module._modules[name] = new_attn

                print(
                    f"Converted {full_name}: "
                    f"layer_idx={layer_idx}, "
                    f"target_device={target_device}, "
                    f"new_device={next(new_attn.parameters()).device}"
                )

                layer_idx_counter += 1

    recurse(model)
    return model
