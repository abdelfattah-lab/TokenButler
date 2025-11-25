import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import gc

from typing import Any, Dict, List, Optional, Tuple
import traceback
import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb

from utils import LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding, repeat_kv, sorted_index_to_mask
from transformers.cache_utils import DynamicCache

from triton_kernels.flash_attn import attention
from triton_kernels.flash_attn_mse_loss import attention_mse_loss


class PredictorDynamicCache(DynamicCache):
    def __init__(self):
        super().__init__()
        self.predictor_primary_key: List[Optional[torch.Tensor]] = []
        self.predictor_primary_value: List[Optional[torch.Tensor]] = []
        self.predictor_importance_key: List[Optional[torch.Tensor]] = []

    def update_predictor_primary(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append or create the predictor's "primary" K/V states for `layer_idx`.

        shape for key_states, value_states is typically [batch_size, num_heads, seq_len, head_dim].
        """
        # Extend the lists so that `predictor_primary_key[layer_idx]` and
        # `predictor_primary_value[layer_idx]` exist.
        self._ensure_list_capacity(
            self.predictor_primary_key, layer_idx, fill=None
        )
        self._ensure_list_capacity(
            self.predictor_primary_value, layer_idx, fill=None
        )

        # If this is the very first time we are updating that layer's predictor cache, just assign
        if self.predictor_primary_key[layer_idx] is None:
            self.predictor_primary_key[layer_idx] = key_states
            self.predictor_primary_value[layer_idx] = value_states
        else:
            # Otherwise, concatenate along the seq_len dimension (=-2 or =2 depending on your shape).
            self.predictor_primary_key[layer_idx] = torch.cat(
                [self.predictor_primary_key[layer_idx], key_states], dim=2
            )
            self.predictor_primary_value[layer_idx] = torch.cat(
                [self.predictor_primary_value[layer_idx], value_states], dim=2
            )

        return (
            self.predictor_primary_key[layer_idx],
            self.predictor_primary_value[layer_idx],
        )

    def update_predictor_importance(
        self,
        key_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Append or create the predictor's "importance" key for `layer_idx`.
        """
        self._ensure_list_capacity(
            self.predictor_importance_key, layer_idx, fill=None
        )

        if self.predictor_importance_key[layer_idx] is None:
            self.predictor_importance_key[layer_idx] = key_states
        else:
            self.predictor_importance_key[layer_idx] = torch.cat(
                [self.predictor_importance_key[layer_idx], key_states], dim=-2
            )
        return self.predictor_importance_key[layer_idx]

    @staticmethod
    def _ensure_list_capacity(lst: list, idx: int, fill=None):
        if len(lst) <= idx:
            lst.extend([fill] * (idx + 1 - len(lst)))

    def crop(self, max_length: int):
        super().crop(max_length)
        # Now also crop predictor caches
        for idx in range(len(self.predictor_primary_key)):
            if self.predictor_primary_key[idx] is not None:
                self.predictor_primary_key[idx] = self.predictor_primary_key[idx][..., :max_length, :]
                self.predictor_primary_value[idx] = self.predictor_primary_value[idx][..., :max_length, :]

        for idx in range(len(self.predictor_importance_key)):
            if self.predictor_importance_key[idx] is not None:
                self.predictor_importance_key[idx] = self.predictor_importance_key[idx][..., :max_length, :]

        # Remember to adjust self._seen_tokens accordingly
        self._seen_tokens = min(self._seen_tokens, max_length)

    def batch_split(
        self, full_batch_size: int, split_size: int, num_hidden_layers: int = None
    ) -> List["PredictorDynamicCache"]:
        # Use the base split logic for the standard K/V
        base_splits = super().batch_split(full_batch_size, split_size, num_hidden_layers)
        # `base_splits` is now a list of new DynamicCache objects. But we *actually*
        # want them to be PredictorDynamicCache so we can store the predictor states.
        # Easiest: we can cast and fill them. 
        out: List[PredictorDynamicCache] = []

        for split_i, base_split in enumerate(base_splits):
            # Construct an empty PredictorDynamicCache
            new_cache = PredictorDynamicCache()
            # Copy over the underlying fields from base_split
            new_cache.key_cache = base_split.key_cache
            new_cache.value_cache = base_split.value_cache
            new_cache._seen_tokens = base_split._seen_tokens

            # Now also slice our predictor fields
            # The slice in batch dim is [i:i+split_size].
            b_start = split_i * split_size
            b_end = min(full_batch_size, b_start + split_size)

            new_cache.predictor_primary_key = self._slice_list_tensors(
                self.predictor_primary_key, b_start, b_end
            )
            new_cache.predictor_primary_value = self._slice_list_tensors(
                self.predictor_primary_value, b_start, b_end
            )
            new_cache.predictor_importance_key = self._slice_list_tensors(
                self.predictor_importance_key, b_start, b_end
            )

            out.append(new_cache)

        return out

    @classmethod
    def from_batch_splits(cls, splits: List["PredictorDynamicCache"], num_hidden_layers: int = None) -> "PredictorDynamicCache":
        # Let the base class handle the normal K/V merges
        base_merged = DynamicCache.from_batch_splits(splits, num_hidden_layers=num_hidden_layers)
        merged = cls()
        merged.key_cache = base_merged.key_cache
        merged.value_cache = base_merged.value_cache
        merged._seen_tokens = base_merged._seen_tokens

        # Now unify predictor states by concatenating along batch dim=0
        merged.predictor_primary_key = cls._merge_list_tensors(
            [split.predictor_primary_key for split in splits]
        )
        merged.predictor_primary_value = cls._merge_list_tensors(
            [split.predictor_primary_value for split in splits]
        )
        merged.predictor_importance_key = cls._merge_list_tensors(
            [split.predictor_importance_key for split in splits]
        )

        return merged

    def batch_repeat_interleave(self, repeats: int):
        super().batch_repeat_interleave(repeats)
        self.predictor_primary_key = self._repeat_list_tensors(
            self.predictor_primary_key, repeats
        )
        self.predictor_primary_value = self._repeat_list_tensors(
            self.predictor_primary_value, repeats
        )
        self.predictor_importance_key = self._repeat_list_tensors(
            self.predictor_importance_key, repeats
        )

    def batch_select_indices(self, indices: torch.Tensor):
        super().batch_select_indices(indices)
        self.predictor_primary_key = self._select_list_tensors(
            self.predictor_primary_key, indices
        )
        self.predictor_primary_value = self._select_list_tensors(
            self.predictor_primary_value, indices
        )
        self.predictor_importance_key = self._select_list_tensors(
            self.predictor_importance_key, indices
        )

    @staticmethod
    def _slice_list_tensors(
        tensor_list: List[Optional[torch.Tensor]], start: int, end: int
    ) -> List[Optional[torch.Tensor]]:
        out = []
        for t in tensor_list:
            if t is None:
                out.append(None)
            else:
                out.append(t[start:end, ...])
        return out

    @classmethod
    def _merge_list_tensors(
        cls, list_of_lists: List[List[Optional[torch.Tensor]]]
    ) -> List[Optional[torch.Tensor]]:
        # If no splits, return empty
        if not list_of_lists:
            return []

        # Number of layers is length of the sub-list from the first split
        max_len = len(list_of_lists[0])
        merged = [None] * max_len

        for layer_idx in range(max_len):
            # collect that layer_idx from each split
            chunk_tensors = []
            for split in list_of_lists:
                t = split[layer_idx] if layer_idx < len(split) else None
                if t is not None:
                    chunk_tensors.append(t)
            if len(chunk_tensors) == 0:
                merged[layer_idx] = None
            else:
                merged[layer_idx] = torch.cat(chunk_tensors, dim=0)
        return merged

    @staticmethod
    def _repeat_list_tensors(
        tensor_list: List[Optional[torch.Tensor]], repeats: int
    ) -> List[Optional[torch.Tensor]]:
        out = []
        for t in tensor_list:
            if t is None:
                out.append(None)
            else:
                out.append(t.repeat_interleave(repeats, dim=0))
        return out

    @staticmethod
    def _select_list_tensors(
        tensor_list: List[Optional[torch.Tensor]], indices: torch.Tensor
    ) -> List[Optional[torch.Tensor]]:
        out = []
        for t in tensor_list:
            if t is None:
                out.append(None)
            else:
                out.append(t.index_select(0, indices))
        return out

class TokenImportancePredictorAttentive(nn.Module):
    def __init__(
        self,
        config,
        pred_hid_size,
        num_heads,
        num_hidden_layers,
        dDash,
        intdim,
        attn_reduce_factor,
        dropout: float = 0.1,
        predictor_variant: str = "tokenbutler_project",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = pred_hid_size
        self.num_heads = num_heads
        # Interpreted as the number of slots per producer group.
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.head_dim = pred_hid_size // (num_heads * 4)
        self.rope_theta = config.rope_theta
        self.dDash = dDash
        self.intermediate_dim = intdim
        self.attn_reduce_factor = attn_reduce_factor
        self.max_position_embeddings = config.max_position_embeddings
        self.flash_attn = False
        self.predictor_variant = predictor_variant

        if self.predictor_variant != "tokenbutler_project":
            raise ValueError(
                f"Unsupported predictor_variant: {self.predictor_variant}. Only tokenbutler_project is supported."
            )

        # Real model head dim (for projecting true K cache)
        num_attn_heads = getattr(config, "num_attention_heads", num_heads)
        self.model_head_dim = config.hidden_size // num_attn_heads

        assert (
            pred_hid_size % (num_heads * 4) == 0
        ), "pred_hid_size must be divisible by num_heads * 4."

        # Enforce dDash ≤ head_dim (because we project the real key cache)
        if self.dDash > self.model_head_dim:
            raise ValueError(
                f"dDash={self.dDash} must be <= model head dim={self.model_head_dim} "
                "for tokenbutler_project."
            )

        # Reduced hidden size for the (optional) mini self-attention
        self.hidden_size_reduced = self.hidden_size // self.attn_reduce_factor
        assert (
            self.hidden_size_reduced % self.num_heads == 0
        ), "Reduced hidden size must be divisible by num_heads"
        self.attn_head_dim = self.hidden_size_reduced // self.num_heads

        # Shared LayerNorm for the importance branch
        self.norm_importance = nn.LayerNorm(self.hidden_size)

        # TokenButler Project:
        #   - no internal self-attn
        #   - K comes from the real key cache via a learned projection
        #   - Q-MLP predicts queries for a *group* of layers
        #     (slots within a producer group).
        #   Here, `num_hidden_layers` is interpreted as:
        #       N_slots = producer_frequency = number of consumer layers
        #       served by each producer.
        self.num_query_mlps = 1
        self.layers_per_slot = self.num_hidden_layers  # == N_slots

        # Single MLP:
        #   [B, L, hidden] → [B, L, (N_slots * H * dDash)]
        out_dim_per_slot = self.layers_per_slot * self.num_heads * self.dDash

        self.q_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(pred_hid_size, self.intermediate_dim, bias=False),
                    nn.SiLU(),
                    nn.Linear(
                        self.intermediate_dim,
                        out_dim_per_slot,
                        bias=False,
                    ),
                )
            ]
        )
        # Initialize all modules that actually exist
        self._initialize_weights()
        self.device = None

        # Per-(slot, head) projection of *real* KV cache.
        # Shape: [num_slots, num_heads, head_dim, dDash]
        self.key_cache_proj = nn.Parameter(
            torch.empty(
                self.config.num_hidden_layers,
                self.num_heads,
                self.model_head_dim,
                self.dDash,
            )
        )
        nn.init.xavier_uniform_(self.key_cache_proj)


    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # Xavier initialization for linear layers
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize in_proj_weight
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)

                # Initialize out_proj
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        layer_idx=None,
    ):
        """
        Args:
            hidden_states: [B, L, hidden_size]
        Returns:
            q_importance: [B*H, N_slots, Lq, dDash]
            k_importance: [B*H, N_slots, Lk, dDash] or None
        """
        layer_idx = 0  # single predictor block

        if self.device != hidden_states.device:
            self.device = hidden_states.device
            self.to(self.device)

        B, L, _ = hidden_states.size()
        # tokenbutler_project:
        # a single Q-MLP predicts queries for a *group* of layers.
        # Each forward() call returns queries for N_slots positions
        # within that group.
        base_linear = self.q_mlps[0][0]
        hidden_states = hidden_states.to(base_linear.weight.dtype)

        hidden_for_importance = self.norm_importance(hidden_states)
        B, L, _ = hidden_for_importance.size()
        H = self.num_heads

        N_slots = self.num_hidden_layers  # == layers_per_slot

        mlp = self.q_mlps[0]
        # mlp output: [B, L, N_slots * H * dDash]
        q_flat = mlp(hidden_for_importance)
        q_slot = q_flat.view(B, L, N_slots, H, self.dDash)  # [B, L, N_slots, H, dDash]
        # [B, H, N_slots, L, dDash]
        q_slot = q_slot.permute(0, 3, 2, 1, 4).contiguous()

        # Final shape expected by the attention code: [B*H, N_slots, L, dDash]
        q_importance = q_slot.view(B * H, N_slots, L, self.dDash)
        k_importance = None  # K comes from real key cache (projected externally)
        return q_importance, k_importance


class HeadImportancePredictor(nn.Module):
    def __init__(self, config, pred_hid_size, num_heads, num_hidden_layers, dDash, intdim, \
                 attn_reduce_factor, dropout=0.1):
        """
        Optimized Token Importance Predictor with parallel Q-K projections and simplified mapping.
        
        Args:
            config: Configuration object containing model parameters.
            pred_hid_size (int): Hidden size for the predictor's attention layer.
            num_heads (int): Number of attention heads.
            num_hidden_layers (int): Number of transformer layers to predict.
            dropout (float): Dropout probability.
            q_downscale (int): Factor to downscale the Q dimension for efficiency.
            intermediate_dim (int): Intermediate dimension for non-linear transformations in projections.
        """
        super().__init__()
        self.is_head_predictor = None
        self.config = config
        self.hidden_size = pred_hid_size
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.head_dim = pred_hid_size // (num_heads * 4)
        self.rope_theta = config.rope_theta
        self.dDash = dDash
        self.intermediate_dim = intdim
        self.attn_reduce_factor = attn_reduce_factor
        self.max_position_embeddings = config.max_position_embeddings
        self.flash_attn = False

        # Reduce the hidden size for attention computations
        self.hidden_size_reduced = self.hidden_size // self.attn_reduce_factor  # For example, reduce to 1/4th
        assert self.hidden_size_reduced % self.num_heads == 0, "Reduced hidden size must be divisible by num_heads"
        self.attn_head_dim = self.hidden_size_reduced // self.num_heads

        # Input projection to reduce hidden size
        self.input_proj = nn.Linear(self.hidden_size, self.hidden_size_reduced, bias=False)

        # Query, Key, Value projections for attention
        self.q_proj_attn = nn.Linear(self.hidden_size_reduced, self.hidden_size_reduced, bias=False)
        self.k_proj_attn = nn.Linear(self.hidden_size_reduced, self.hidden_size_reduced, bias=False)
        self.v_proj_attn = nn.Linear(self.hidden_size_reduced, self.hidden_size_reduced, bias=False)
        # Output projection to restore hidden size
        # self.o_proj_attn = nn.Linear(self.hidden_size_reduced, self.hidden_size_reduced, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout)

        # LayerNorm and Feed-forward network
        self.norm1 = nn.LayerNorm(self.hidden_size_reduced)
        self.norm2 = nn.LayerNorm(self.hidden_size)

        self.ffn_hidden_size = 4 * self.hidden_size_reduced  # Typical FFN hidden size
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size_reduced, self.ffn_hidden_size),
            nn.GELU(),
            nn.Linear(self.ffn_hidden_size, self.num_heads * self.num_hidden_layers),
        )

        # Initialize rotary positional embeddings
        self._init_rope()
        self._initialize_weights()
        self.device = None

    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # Xavier initialization for linear layers
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize in_proj_weight
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)

                # Initialize out_proj
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)

    def _init_rope(self):
        config_copy = copy.deepcopy(self.config)
        config_copy.head_dim = self.attn_head_dim
        # Rotary embedding for attention layer
        self.rotary_emb_attn = LlamaRotaryEmbedding(
            config_copy
        )
        # Rotary embedding for importance projection
        self.rotary_emb_importance = LlamaRotaryEmbedding(
            config_copy
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        """
        Forward pass for the Optimized Token Importance Predictor.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [B, L, HQ].
            attention_mask (torch.Tensor, optional): Attention mask of shape [B, 1, 1, L] or [B, 1, L, L].
            position_ids (torch.Tensor, optional): Position IDs.
            past_key_value (tuple, optional): Past key and value states.
            use_cache (bool, optional): Whether to use cache.
        
        Returns:
            torch.Tensor: Importance scores of shape [B, N, H, L, L].
        """
        if self.device != hidden_states.device:
            self.device = hidden_states.device
            self.to(self.device)

        B, L, E = hidden_states.size()
        if past_key_value is None:
            past_key_value = {}
        past_primary = past_key_value.get('primary', None)
        # Reduce hidden size
        hidden_states = hidden_states.to(self.input_proj.weight.dtype)
        hidden_states_reduced = self.input_proj(hidden_states)
        # Compute q, k, v for attention
        q = self.q_proj_attn(hidden_states_reduced)
        k = self.k_proj_attn(hidden_states_reduced)
        v = self.v_proj_attn(hidden_states_reduced)
        q = q.view(B, L, self.num_heads, self.attn_head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.attn_head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.attn_head_dim).transpose(1, 2)
        if past_primary is not None:
            past_L = past_primary[0].shape[2]
            kv_seq_len = past_L + L
        else:
            kv_seq_len = L
        
        cos, sin = self.rotary_emb_attn(v, position_ids)
        if position_ids is None:
            position_ids = torch.arange(kv_seq_len, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(B, kv_seq_len)
        
        if past_primary is not None:
            k = torch.cat([past_primary[0], k], dim=2)
            v = torch.cat([past_primary[1], v], dim=2)
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        if use_cache:
            past_key_value['primary'] = (k.detach(), v.detach())

        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        attn_output = attn_output.to(q.dtype)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.hidden_size_reduced)
        attn_output = self.norm1(attn_output)
        head_importances = self.ffn(attn_output)
        return head_importances, past_key_value
