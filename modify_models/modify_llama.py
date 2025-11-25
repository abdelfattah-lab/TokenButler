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

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb

from utils import LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding, repeat_kv, sorted_index_to_mask, sorted_index_to_variable_mask
from utils import calculate_hit_metrics, calculate_effective_sparsity, threshold_to_mask, SlidingWindowCache, enforce_sliding_window
from transformers.cache_utils import DynamicCache
from predictor import TokenImportancePredictorAttentive, PredictorDynamicCache, HeadImportancePredictor, attention_mse_loss, attention

from triton_kernels.flash_attn import attention
from triton_kernels.flash_attn_mse_loss import attention_mse_loss
import os, csv, hashlib
# torch.backends.cuda.enable_flash_sdp(enabled=True)
# torch.backends.cuda.enable_mem_efficient_sdp(enabled=True)

class LlamaAttentionExperimental(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        producer: Optional["LlamaAttentionExperimental"] = None,
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

        # Layer / grouping metadata
        self.layer_idx = layer_idx

        # How many *consumer* layers each predictor invocation serves.
        # Example (24‑layer model, producer_frequency=4):
        #   layer 0 → queries for layers 1–4
        #   layer 4 → queries for layers 5–8
        #   layer 8 → queries for layers 9–12
        self.producer_frequency = max(1, int(producer_frequency))

        # Module that owns the predictor outputs for this layer.
        # For layer 0 this stays None.
        self.producer: Optional["LlamaAttentionExperimental"] = producer

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
        # Each predictor call produces one query tensor per consumer layer in
        # the group, so we set num_layers_pred = producer_frequency.
        self.num_layers_pred = self.producer_frequency
        self.num_tok_per_page = None
        self.calc_hitrates = False
        self.flash_attn = False
        # --- predictor long-context training knobs ---
        # Maximum number of query positions (rows) used for the aux loss.
        # Set to None or 0 to fall back to full L^2 (old behavior).
        self.max_loss_rows = 256
        # Fraction of the sequence tail to sample from (we always include the last token).
        self.loss_tail_fraction = 0.5
        self.train_headpredictor = False
        self.calibrate_thresholds = False
        self.test_with_thresholds = False
        self.late_context_upweight = False
        self.softmax_causal_loss_mse = False
        self.softmax_causal_loss_ce = False
        self.old_predictor = None
        self.pairwise_loss = False
        self.mode = "balanced"  # "extreme_recall" or "balanced"
        self.pairwise_topk_ratio = 0.02  # fraction of keys used as pos/neg in pairwise loss
        self.tokenbutler_variant = "tokenbutler"

        # Whether this layer runs the predictor (i.e. is the group root that
        # produces queries for producer_frequency future layers).
        self.is_predictor_owner = is_predictor_owner
        # --- sparsity control knobs (decode‑time only) ---
        # For fixed_xpc: target_sparsity is fraction of *candidate* tokens pruned (0‑1).
        # For fixed_ytok: target_keep_tokens is the number of *candidate* tokens kept.
        self.target_sparsity = None
        self.target_keep_tokens = None
        # Decode‑time gating: prefill + first generated token stay dense.
        self._dense_kv_cutoff = 0
        self.always_dense_decode_tokens = 1

        if self.mode == "extreme_recall": # top4
            self.low_recall_first = {   }
        elif self.mode == "balanced": # top50
            self.low_recall_first = { }
        else:
            raise ValueError(f"Unknown sparsity mode {self.mode}")

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
            # This module stores the predictor outputs for its consumer group.
            self.q_importance = None  # [BH, producer_frequency, L, dDash]
            self.k_importance = None
            self.head_importances = None
            self.actmagn_masklist = {}
            self.available_tokens = {}

        # Attention setup
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _compute_global_head_keep(self):
        """
        Build a tensor  shape [num_hidden_layers, num_heads] that contains
        the fraction of keys to keep for *every* (layer, head) pair,
        scaled so the **model‑wide** average equals self.sparse_aggression.
        Called once, cached in self._global_head_keep.
        """
        L, H = self.num_hidden_layers, self.num_heads
        keep = torch.full((L, H), float(self.sparse_aggression))

        # lowrecall_tuples is already ordered: earlier ⇒ worse recall
        bad_pairs = self.lowrecall_tuples
        if not bad_pairs:                    # no calibration data
            self._global_head_keep = keep
            return

        keep_max, keep_min = 1.0, float(self.sparse_aggression)
        N = len(bad_pairs)
        for rank, (head_idx, layer_idx) in enumerate(bad_pairs):
            frac = rank / (N - 1 + 1e-5)     # 0 … 1
            keep[layer_idx, head_idx] = keep_max - frac * (keep_max - keep_min)

        # --- global renormalisation ---------------------------------
        total_heads = L * H
        scale = (self.sparse_aggression * total_heads) / keep.sum()
        keep *= scale
        keep.clamp_(max=1.0)

        self._global_head_keep = keep        # cache

    # ----------------------------------------------------------------
    def build_head_keep_ratios(self):
        """
        Return the [num_heads] vector for *this* layer, using
        model‑global calibration.  Safe to call every forward(); the
        table is computed once and cached.
        """
        if not hasattr(self, "_global_head_keep"):
            self._compute_global_head_keep()
        return self._global_head_keep[self.layer_idx].to(next(self.parameters()).device)

    def update_predictor(self):
        # Device of this attention block (respects device_map / accelerate)
        attn_device = next(self.parameters()).device

        self.sparse_token_predictor = TokenImportancePredictorAttentive(
            self.config,
            self.pred_hid_size,
            self.num_heads,
            self.num_layers_pred,
            dDash=self.dDash,
            intdim=self.intdim,
            attn_reduce_factor=self.attn_reduce_factor,
            dropout=0.1,
            predictor_variant=getattr(self, "tokenbutler_variant", "tokenbutler"),
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
            ).to(attn_device)  # ⬅️ was '.to("cuda:0")' before
            self.sparse_head_predictor.flash_attn = self.flash_attn

    def set_token_sparsity(self):
        assert self.token_sparse_method is not None, "Set token sparse method first!"
        method = self.token_sparse_method

        # Optional: load per‑head threshold calibration if available.
        if method is not None:
            try:
                mname = self.config._name_or_path.split("/")[-1]
                read_path = f"threshold_calibs/{mname}/{method}.pkl"
                threshold_model_dictionary = torch.load(read_path)
                self.tok_calibration_set = threshold_model_dictionary
            except Exception:
                pass

        # Reset sparsity state for this layer.
        self.target_sparsity = None
        self.target_keep_tokens = None
        self.sparse_aggression = None
        self.head_keep = None

        if method.startswith("fixed_"):
            # fixed_xpc / fixed_ytok
            spec = method.split("_", 1)[1]

            # fixed_xpc: x is *sparsity* (fraction of candidate tokens pruned).
            if spec.endswith("pc"):
                if self.layer_idx == 0:
                    # Never prune layer 0.
                    self.target_sparsity = 0.0
                    self.sparse_aggression = 1.0
                else:
                    x = float(spec[:-2])
                    self.target_sparsity = x / 100.0              # prune fraction on candidates
                    self.sparse_aggression = 1.0 - self.target_sparsity  # keep fraction on candidates

                # Per‑head keep ratios, globally renormalised to the keep fraction.
                self.head_keep = self.build_head_keep_ratios()

            # fixed_ytok: keep exactly y *candidate* tokens per head/query.
            elif spec.endswith("tok"):
                y = int(spec[:-3])

                if self.layer_idx == 0:
                    # Never prune layer 0.
                    self.target_keep_tokens = None
                    self.sparse_aggression = 1.0
                else:
                    self.target_keep_tokens = max(1, y)

                    # --- Approximate keep fraction for logging / metrics only ----
                    # We'll approximate the effective number of candidate tokens
                    # using either a user‑provided nominal_seq_len or the model's
                    # max_position_embeddings, minus sink + sliding window.
                    nominal_L = getattr(self, "nominal_seq_len", None)
                    if nominal_L is None:
                        nominal_L = getattr(
                            self.config,
                            "max_position_embeddings",
                            self.max_position_embeddings,
                        )

                    head = getattr(self, "min_sparse_index", 0) or 0
                    tail = getattr(self, "sliding_window", 0) or 0
                    eff_L = max(1, nominal_L - head - tail)

                    keep_frac = min(1.0, float(self.target_keep_tokens) / eff_L)

                    # This is only used by configure_experimental_modules() to
                    # compute an "average sparsity" summary.
                    self.sparse_aggression = keep_frac
                    self.head_keep = None
            else:
                raise ValueError(f"Unknown fixed sparsity spec '{spec}' in token_sparse_method='{method}'")
        else:
            # We no longer support LazyLLM / progressive schemes here.
            raise ValueError(
                f"Unsupported token sparsity method '{method}'. "
                "Use 'fixed_xpc' (e.g. fixed_65pc) or 'fixed_ytok' (e.g. fixed_128tok)."
            )

    def _should_apply_sparse_decode(self, q_len: int, kv_seq_len: int) -> bool:
        """
        Decode‑time gating:
          - Prefill (q_len > 1) is always dense.
          - The *first* decode token after prefill is also dense.
          - Pruning starts from the second decode step.
        """
        if q_len != 1:
            # Only single‑token decode steps are sparsified.
            return False

        # If we never saw a prefill for this sequence, treat the first decode
        # call as dense and remember its kv_seq_len as the cutoff.
        if self._dense_kv_cutoff == 0:
            self._dense_kv_cutoff = kv_seq_len
            return False

        # After prefill, forward() sets _dense_kv_cutoff = prefill_len + 1.
        # We start pruning once kv_seq_len grows beyond that.
        return kv_seq_len > self._dense_kv_cutoff

    def _build_decode_mask_fixed(
        self,
        importance_scores: torch.Tensor,
        attention_mask: torch.Tensor,
        min_sparse_index: Optional[int],
    ) -> torch.Tensor:
        """
        Build a decode‑time sparsity mask for fixed_xpc / fixed_ytok.

        importance_scores: [B, H, 1, K] – predictor softmax over keys.
        attention_mask:    [B, 1, 1, K] – 0 or -inf.

        Returns:
            mask_tensor: [B, H, 1, K] with values in {0, -inf}, where -inf marks
            tokens pruned by TokenButler (excluding sink + sliding‑window tokens).
        """
        bsz, num_heads, q_len, key_len = importance_scores.shape
        assert q_len == 1, "Decode‑time mask only makes sense for q_len == 1"
        device = importance_scores.device
        dtype = importance_scores.dtype

        # Candidate tokens:
        #   - not masked by attention_mask
        #   - not in the sink region [0:min_sparse_index)
        #   - not in the sliding‑window tail (always‑keep region)
        attn_valid = (attention_mask[:, :, -1:, :] == 0)  # [B,1,1,K]
        candidate_mask = attn_valid.expand(bsz, num_heads, 1, key_len)

        if min_sparse_index is not None and min_sparse_index > 0:
            clamp_idx = min(min_sparse_index, key_len)
            candidate_mask[..., :clamp_idx] = False

        if self.sliding_window is not None and self.sliding_window > 0:
            win = min(self.sliding_window, key_len)
            candidate_mask[..., -win:] = False

        if not candidate_mask.any():
            # Nothing we are allowed to drop – stay dense.
            return torch.zeros_like(importance_scores, dtype=dtype, device=device)

        method = self.token_sparse_method or ""
        candidate_counts = candidate_mask.sum(dim=-1, keepdim=True)  # [B,H,1,1]

        # How many *candidate* tokens do we keep per (B, H, row)?
        if "pc" in method:
            if self.sparse_aggression is None:
                raise ValueError("sparse_aggression must be set for fixed_xpc")
            # Per‑head keep ratios [H] with global normalisation.
            if getattr(self, "head_keep", None) is not None:
                head_keep = self.head_keep.to(device)
            else:
                head_keep = torch.full(
                    (self.num_heads,), float(self.sparse_aggression), device=device
                )
            head_keep = head_keep.clamp(min=0.0, max=1.0).view(1, self.num_heads, 1, 1)
            keep_counts = (head_keep * candidate_counts.float()).floor()
            keep_counts = keep_counts.clamp(min=1)
            keep_counts = torch.minimum(keep_counts, candidate_counts)
            keep_counts = keep_counts.long()

        elif "tok" in method:
            if self.target_keep_tokens is None or self.target_keep_tokens <= 0:
                return torch.zeros_like(importance_scores, dtype=dtype, device=device)
            y = self.target_keep_tokens
            keep_counts = torch.minimum(
                candidate_counts,
                torch.full_like(candidate_counts, y, dtype=torch.long),
            )
            keep_counts = keep_counts.clamp(min=1)
        else:
            raise ValueError(f"token_sparse_method '{method}' is not a fixed_* scheme")

        # Rank candidate tokens by importance (descending), ignoring non‑candidates.
        scores = importance_scores.clone()
        scores = scores.masked_fill(~candidate_mask, float("-inf"))
        _, sorted_idx = scores.sort(dim=-1, descending=True)  # [B,H,1,K]

        # # Invert the sort to get per‑key rank.
        # B, H, _, K = sorted_idx.shape
        # arange_K = torch.arange(K, device=device).view(1, 1, 1, K)
        # rank = torch.empty_like(sorted_idx)
        # rank.scatter_(-1, sorted_idx, arange_K)
        B, H, q_len, key_len = sorted_idx.shape
        # rank[b, h, q, j] will hold the rank position of key j for that (b,h,q)
        rank = torch.empty_like(sorted_idx, dtype=torch.long)
        # Build src with the same shape as the index tensor along non-scatter dims
        arange_K = torch.arange(key_len, device=sorted_idx.device, dtype=torch.long)
        arange_K = arange_K.view(1, 1, 1, key_len).expand_as(sorted_idx)  # [B,H,1,K]
        rank.scatter_(-1, sorted_idx, arange_K)
        # Keep:
        #   - all non‑candidate tokens, plus
        #   - the top 'keep_counts' candidate tokens.
        keep_mask = (~candidate_mask) | (rank < keep_counts)

        mask_tensor = torch.zeros_like(importance_scores, dtype=dtype, device=device)
        mask_tensor = mask_tensor.masked_fill(~keep_mask, float("-inf"))

        # Explicitly clear sink + sliding‑window regions (always keep).
        if min_sparse_index is not None and min_sparse_index > 0:
            clamp_idx = min(min_sparse_index, key_len)
            mask_tensor[..., :clamp_idx] = 0.0
        if self.sliding_window is not None and self.sliding_window > 0:
            win = min(self.sliding_window, key_len)
            mask_tensor[..., -win:] = 0.0

        return mask_tensor


    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.config
            )
        else:
            scaling_type = self.config.rope_scaling.get("type") or self.config.rope_scaling.get("rope_type")
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear" or scaling_type == 'llama3':
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    config=self.config
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    config=self.config
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _get_group_slot_index(self) -> Optional[int]:
        """
        Local slot index into the producer's query tensor for this layer,
        in [0, producer_frequency-1].

        With producer_frequency = G:
          layer 1 → slot 0, 2 → slot 1, …, G → slot G-1,
          layer G+1 → slot 0 (served by producer at layer G), etc.

        Returns None for layer 0 or if no producer is configured.
        """
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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[PredictorDynamicCache]]:
        bsz, q_len, _ = hidden_states.size()

        Ltrack = hidden_states.size(1)
        attn_device = self.q_proj.weight.device

        if hidden_states.device != attn_device:
            hidden_states = hidden_states.to(attn_device)
        if attention_mask is not None and attention_mask.device != attn_device:
            attention_mask = attention_mask.to(attn_device)
        if position_ids is not None and position_ids.device != attn_device:
            position_ids = position_ids.to(attn_device)
        if padding_mask is not None and padding_mask.device != attn_device:
            padding_mask = padding_mask.to(attn_device)
        if cache_position is not None and cache_position.device != attn_device:
            cache_position = cache_position.to(attn_device)

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        evalmode = self.eval_llm_mode
        # query_states/key_states/value_states will be reshaped to [B, H, L, D].
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        cos = cos.to(query_states.device)
        sin = sin.to(query_states.device)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        if use_cache:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        kv_seq_len = key_states.shape[-2]

        # Track where we are in the sequence for decode‑time sparsity.
        # Heuristic: a call with q_len > 1 and kv_seq_len == q_len is a full prefill
        # for a fresh sequence.  We then keep this prefill *and* the first decode
        # token dense, and only apply TokenButler pruning afterwards.
        if self.inference_mode and use_cache and q_len > 1 and kv_seq_len == q_len:
            # New sequence prefill: remember the length and give the next
            # 'always_dense_decode_tokens' decode steps dense attention.
            self._dense_kv_cutoff = kv_seq_len + self.always_dense_decode_tokens
        final_mask = None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_len = key_states.size(2)
        bsz, q_len = query_states.size(0), query_states.size(2)

        if attention_mask is None:
            causal_mask_2d = torch.ones(q_len, kv_seq_len, 
                                        device=hidden_states.device, 
                                        dtype=torch.bool).triu(diagonal=1)
            causal_mask_4d = causal_mask_2d.unsqueeze(0).expand(bsz, 1, q_len, kv_seq_len)
            attention_mask = torch.full_like(causal_mask_4d, 0, dtype=hidden_states.dtype)
            if q_len != 1:
                attention_mask = attention_mask.masked_fill(causal_mask_4d, float("-inf"))
                
        if self.inference_mode:
            min_sparse_index = self.min_sparse_index
            with torch.no_grad():
                if evalmode == "ExpPred":
                    # Only run predictor-based sparsity on *single-token decode* steps.
                    # Prefill (q_len > 1) stays dense to avoid [B,H,Lq,Lk] importance mats.
                    if self.layer_idx > 0 and q_len == 1:
                        # ------------------------------------------------------------------
                        # TokenButler variants:
                        #   tokenbutler         → q,k from predictor (original)
                        #   tokenbutler_slice   → q from predictor, k = first dDash dims of real key cache
                        #   tokenbutler_project → q from predictor, k = Linear(real key cache)
                        # ------------------------------------------------------------------
                        # q_importance is stored on the group root as
                        # [BH, producer_frequency, Lq, dDash]. Each layer ℓ>0 uses
                        # the slot corresponding to
                        #   slot_idx = (ℓ - 1) % producer_frequency.
                        slot_idx = self._get_group_slot_index()
                        q_importance_tensor = self.producer.q_importance[
                            :, slot_idx, :, :
                        ].float().to(query_states.device)  # [BH, Lq, dDash]
 
                        if self.tokenbutler_variant == "tokenbutler":
                            k_importance_tensor = self.producer.k_importance[
                                :, slot_idx, :, :
                            ].float().to(key_states.device)  # [BH, Lk, dDash]
                        elif self.tokenbutler_variant == "tokenbutler_slice":
                            Bk, Hk, Lk, Dh = key_states.shape  # [B, H, Lk, head_dim]

                            # Use the projection corresponding to the *actual* layer index,
                            # NOT the predictor slot.
                            proj_weight = self.producer.sparse_token_predictor.key_cache_proj[
                                self.layer_idx
                            ]  # [H, Dh, dDash]

                            key_for_proj = key_states.to(proj_weight.device, dtype=proj_weight.dtype)  # [B,H,Lk,Dh]
                            k_proj = torch.einsum(
                                "bhlk,hkd->bhld", key_for_proj, proj_weight
                            )  # [B,H,Lk,dDash]
                            if k_proj.device != key_states.device:
                                k_proj = k_proj.to(key_states.device)

                            k_importance_tensor = k_proj.reshape(Bk * Hk, Lk, self.dDash)
                        elif self.tokenbutler_variant == "tokenbutler_project":
                            Bk, Hk, Lk, Dh = key_states.shape  # [B, H, Lk, head_dim]
                            # One projector per *real* layer & head.
                            proj_weight = self.producer.sparse_token_predictor.key_cache_proj[
                                self.layer_idx
                            ]  # [H, Dh, dDash]

                            # Move keys to predictor weight device, do projection there, then move back.
                            key_for_proj = key_states.to(proj_weight.device, dtype=proj_weight.dtype)            # [B,H,Lk,Dh]
                            k_proj = torch.einsum(
                                "bhlk,hkd->bhld", key_for_proj, proj_weight
                            )  # [B,H,Lk,dDash]
                            if k_proj.device != key_states.device:
                                k_proj = k_proj.to(key_states.device)

                            k_importance_tensor = k_proj.reshape(Bk * Hk, Lk, self.dDash)
                        else:
                            raise ValueError(
                                f"Unknown tokenbutler_variant: {self.tokenbutler_variant}"
                            )

                        importance_mask = torch.bmm(
                            q_importance_tensor,
                            k_importance_tensor.transpose(-2, -1),
                        ) / math.sqrt(self.dDash)  # [BH, Lq, Lk]
                        importance_mask = importance_mask.view(
                            bsz, self.num_heads, q_len, key_len
                        )  # [B, H, Lq, Lk]
                        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                        if self.calc_hitrates:
                            estimated = nn.functional.softmax(importance_mask + attention_mask, dim=-1)
                            truth     = nn.functional.softmax(attn_weights     + attention_mask, dim=-1)
                            head_calibration_stage = False
                            if head_calibration_stage:
                                def topk_recall(pred_scores, true_scores, k_tokens):
                                    topk_pred  = pred_scores.topk(k_tokens, dim=-1).indices      # [..., K]
                                    topk_true  = true_scores.topk(k_tokens, dim=-1).indices      # [..., K]
                                    hits = (topk_pred.unsqueeze(-1) == topk_true.unsqueeze(-2)).any(-1).float()
                                    return hits.mean().item()
                                ratios   = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.8, 0.9]
                                k_vals   = [max(1, int(r * key_len)) for r in ratios]   # guarantee ≥ 1
                                recalls  = {f"top{int(r*100)}": topk_recall(estimated[:, :, -1:, :], truth[:, :, -1:, :], k)
                                            for r, k in zip(ratios, k_vals)}
                                # we only do 6 batch-item calibration.
                                # csv_path = "l3_8b_calibs.csv"
                                csv_path = "l3_8b_calibs.csv"
                                file_exists = os.path.isfile(csv_path)

                                # Stable, deterministic tag for this batch (8‑char SHA‑1 of the query tokens)
                                batch_hash = hashlib.sha1(query_states.detach()
                                                        .cpu().numpy().tobytes()).hexdigest()[:8]

                                fieldnames = ["batch_hash", "layer_idx", "head_idx"]
                                for r, k in zip(ratios, k_vals):
                                    fieldnames.append(f"top{int(r*100)}")

                                with open(csv_path, "a", newline="") as csvfile:
                                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                    if not file_exists:                       # write header once
                                        writer.writeheader()

                                    # ── per‑head recall@K ────────────────────────────────────────────────
                                    for h in range(self.num_heads):
                                        est_h = estimated[:, h:h+1, :, :]     # keep dims for broadcasting
                                        tru_h = truth    [:, h:h+1, :, :]

                                        per_head = {}
                                        for r, k in zip(ratios, k_vals):      # ratios & k_vals already defined
                                            score = topk_recall(est_h, tru_h, k)
                                            per_head[f"top{int(r*100)}"] = score

                                        writer.writerow({
                                            "batch_hash": batch_hash,
                                            "layer_idx": self.layer_idx,
                                            "head_idx":  h,
                                            **per_head
                                        })
                                # ─────────────────────────────────────────────────────────────────────────
                            # Standard token hit accuracy: overlap of teacher/predictor
                            # top-50% keys.
                            self.tok_hit_acc, self.tok_mean_rank_corr, self.tok_max_rank_corr = calculate_hit_metrics(
                                estimated_importance=estimated,
                                true_importance=truth,
                                top_k_ratio=0.5,
                            )
                            # Hard-token diagnostics: how well we recover teacher
                            # tokens above the 90th / 95th percentile (top 10% / 5%).
                            self.tok_hit_acc_90, _, _ = calculate_hit_metrics(
                                estimated_importance=estimated,
                                true_importance=truth,
                                top_k_ratio=0.1,   # top 10% of keys
                            )
                            self.tok_hit_acc_95, _, _ = calculate_hit_metrics(
                                estimated_importance=estimated,
                                true_importance=truth,
                                top_k_ratio=0.05,  # top 5% of keys
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
                            # Main inference path: dense prefill + dense first decode
                            # token, then fixed sparsity on later decode tokens.
                            importance_scores = torch.softmax(importance_mask + attention_mask, dim=-1)

                            apply_sparse = (
                                self.token_sparse_method is not None
                                and self._should_apply_sparse_decode(q_len, kv_seq_len)
                            )

                            if not apply_sparse:
                                # Either prefill or first decode step → no TokenButler pruning.
                                mask_tensor = torch.zeros_like(importance_scores)
                            else:
                                if q_len != 1:
                                    # Safety net: we never sparsify multi‑token blocks in decode mode.
                                    mask_tensor = torch.zeros_like(importance_scores)
                                else:
                                    mask_tensor = self._build_decode_mask_fixed(
                                        importance_scores,
                                        attention_mask,
                                        min_sparse_index,
                                    )

                        # Apply per‑row sliding‑window after sparsity selection
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
                    else:
                        # Layer 0: dense attention only – no predictor-based sparsity mask.
                        pass
                else:
                    raise ValueError(f"Unknown eval mode {evalmode}")

            # Build the final additive mask actually used for attention:
            # causal/padding from `attention_mask` plus (for decode steps)
            # the predictor‑based sparsity mask. We only apply TokenButler
            # sparsity on single‑token decode steps (q_len == 1), matching
            # the old behaviour.
            if attention_mask is not None:
                if final_mask is not None and q_len == 1:
                    full_mask = attention_mask + final_mask
                else:
                    full_mask = attention_mask
            else:
                # No base mask provided; optionally keep only the predictor
                # mask on single‑token decode steps.
                full_mask = final_mask if (final_mask is not None and q_len == 1) else None

            # Use scaled_dot_product_attention so we don't materialize the
            # full [B, H, Lq, Lk] matrix for the V‑product.
            
            if full_mask is not None:
                full_mask = full_mask.to(dtype=query_states.dtype)
            if full_mask is not None:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=full_mask,
                    is_causal=False,  # causality is encoded in full_mask
                )
            else:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=None,
                    is_causal=True,
                )
            attn_output = attn_output.to(query_states.dtype)

            # In inference mode we only build full attention weights if the
            # caller explicitly asks for them.
            attn_weights = None
            if output_attentions:
                logits = torch.matmul(
                    query_states, key_states.transpose(-2, -1)
                ) / math.sqrt(self.head_dim)
                if full_mask is not None:
                    logits = logits + full_mask
                attn_weights = nn.functional.softmax(
                    logits, dim=-1, dtype=torch.float32
                ).to(value_states.dtype)

        else:
            if self.flash_attn:
                raise NotImplementedError("Flash attention with TokenButler is not implemented yet.")
            else:
                # --- Main model attention path: use scaled_dot_product_attention to avoid L^2 materialization ---
                # query_states/key_states/value_states: [B, H, Lq/Lk, D]
                if attention_mask is not None:
                    attention_mask = attention_mask.to(dtype=query_states.dtype)
                if attention_mask is not None:
                    # attention_mask is already 0 / -inf and includes causal + padding
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        query_states,
                        key_states,
                        value_states,
                        attn_mask=attention_mask,
                        is_causal=False,
                    )
                else:
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        query_states,
                        key_states,
                        value_states,
                        attn_mask=None,
                        is_causal=True,
                    )
                attn_output = attn_output.to(query_states.dtype)
    
                attn_weights = None  # we only build full attn_weights if output_attentions=True
                if output_attentions:
                    # EXPENSIVE: only used when explicitly requested
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    if attention_mask is not None:
                        attn_weights = attn_weights + attention_mask
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
                # --- Auxiliary predictor loss: row-subsampled teacher vs predictor logits ---

                if self.layer_idx > 0:
                    # Predictor Q for this *real* layer: [BH, Lq, dDash]
                    # q_importance is stored on the group root as
                    # [BH, producer_frequency, Lq, dDash], where the second
                    # dimension is the slot within the group.
                    slot_idx = self._get_group_slot_index()
                    q_importance_tensor = self.producer.q_importance[
                        :, slot_idx, :, :
                    ].float().to(query_states.device)
                    BH, Lq_imp, _ = q_importance_tensor.shape
                    assert Lq_imp == q_len
                    assert BH == bsz * self.num_heads

                    # Reshape predictor Q to [B, H, Lq, dDash]
                    q_imp = q_importance_tensor.view(
                        bsz, self.num_heads, q_len, self.dDash
                    )

                    # Predictor K depends on TokenButler variant
                    if self.tokenbutler_variant == "tokenbutler":
                        Bk, Hk, Lk, Dh = key_states.shape  # [B, H, Lk, head_dim]

                        # Use the projection corresponding to the *actual* layer index,
                        # NOT the predictor slot.
                        proj_weight = self.producer.sparse_token_predictor.key_cache_proj[
                            self.layer_idx
                        ]  # [H, Dh, dDash]

                        key_for_proj = key_states.to(proj_weight.device, dtype=proj_weight.dtype)  # [B,H,Lk,Dh]
                        k_proj = torch.einsum(
                            "bhlk,hkd->bhld", key_for_proj, proj_weight
                        )  # [B,H,Lk,dDash]
                        if k_proj.device != key_states.device:
                            k_proj = k_proj.to(key_states.device)

                        k_importance_tensor = k_proj.reshape(Bk * Hk, Lk, self.dDash)
                        _, Lk_imp, _ = k_importance_tensor.shape
                        assert Lk_imp == key_len
                        k_imp = k_importance_tensor.view(
                            bsz, self.num_heads, key_len, self.dDash
                        )
                    elif self.tokenbutler_variant == "tokenbutler_slice":
                        if self.dDash > self.head_dim:
                            raise ValueError(
                                f"dDash={self.dDash} > head_dim={self.head_dim} for tokenbutler_slice"
                            )
                        # Use real key cache: [B,H,Lk,Dh] → [B,H,Lk,dDash]
                        k_imp = key_states[..., : self.dDash].to(q_imp.dtype)
                    elif self.tokenbutler_variant == "tokenbutler_project":
                        Bk, Hk, Lk, Dh = key_states.shape  # [B,H,Lk,Dh]

                        # Again: per *real* layer projection, no slots here.
                        proj_weight = self.producer.sparse_token_predictor.key_cache_proj[
                            self.layer_idx
                        ]  # [H, Dh, dDash]

                        key_for_proj = key_states.to(proj_weight.device, dtype=proj_weight.dtype)  # [B,H,Lk,Dh]
                        k_proj = torch.einsum(
                            "bhlk,hkd->bhld", key_for_proj, proj_weight
                        )  # [B,H,Lk,dDash]
                        if k_proj.device != key_states.device:
                            k_proj = k_proj.to(key_states.device)

                        k_imp = k_proj.to(q_imp.dtype)  # [B,H,Lk,dDash]
                    else:
                        raise ValueError(
                            f"Unknown tokenbutler_variant: {self.tokenbutler_variant}"
                        )
                    # --- derive per-example real lengths from attention_mask if possible ---
                    if attention_mask is not None and attention_mask.size(-2) == q_len:
                        # attention_mask: [B, 1, Lq, Lk], values 0 (keep) or -inf (mask)
                        # A query row is "valid" if it has at least one non-masked key.
                        row_valid = (attention_mask[:, 0, :, :] > -1e4).any(dim=-1)  # [B, Lq] bool
                        real_lengths = row_valid.sum(dim=-1)  # [B], number of valid query tokens per example
                    else:
                        # No padding info -> assume all positions are real
                        real_lengths = torch.full(
                            (bsz,),
                            q_len,
                            device=query_states.device,
                            dtype=torch.long,
                        )

                    # To keep it simple and safe, only use rows that are valid for *all* examples:
                    global_max_valid = int(real_lengths.min().item())  # min over batch
                    if global_max_valid <= 0:
                        # Degenerate case: nothing valid; just bail out on aux loss this step
                        self.msemagn_loss = torch.tensor(0.0, device=query_states.device)
                        # (optionally return here)
                    else:
                        effective_q_len = global_max_valid
                    # --- choose query rows for the loss ---
                    if getattr(self, "max_loss_rows", None) is None or self.max_loss_rows <= 0 or self.max_loss_rows >= effective_q_len:
                        loss_row_idx = torch.arange(effective_q_len, device=query_states.device)
                    else:
                        device = query_states.device
                        tail_frac = getattr(self, "loss_tail_fraction", 0.5)
                        tail_len = max(1, int(effective_q_len * tail_frac))
                        tail_start = max(0, effective_q_len - tail_len)
                        tail_range = torch.arange(tail_start, effective_q_len - 1, device=device)  # exclude very last

                        num_extra = max(0, self.max_loss_rows - 1)
                        if num_extra > 0 and tail_range.numel() > 0:
                            rand_idx = tail_range[torch.randint(0, tail_range.numel(), (num_extra,), device=device)]
                            last_idx = torch.tensor([effective_q_len - 1], device=device)
                            loss_row_idx = torch.cat([rand_idx, last_idx])
                        else:
                            loss_row_idx = torch.tensor([effective_q_len - 1], device=device)

                        loss_row_idx = torch.unique(loss_row_idx)
    
                    R = loss_row_idx.numel()  # number of rows used for aux loss
    
                    # Teacher logits on sub-sampled rows: [B,H,R,Lk]
                    Q_teacher_sub = query_states[:, :, loss_row_idx, :]  # [B,H,R,D]
                    K_teacher_full = key_states                          # [B,H,Lk,D]
                    teacher_logits = torch.matmul(
                        Q_teacher_sub,
                        K_teacher_full.transpose(-2, -1)
                    ) / math.sqrt(self.head_dim)
    
                    # Slice attention_mask to match [B,1,R,Lk] if it has per-row dim.
                    attn_mask_sub = None
                    if attention_mask is not None:
                        if attention_mask.size(-2) == q_len:
                            attn_mask_sub = attention_mask[:, :, loss_row_idx, :]  # [B,1,R,Lk]
                        else:
                            # e.g., [B,1,1,Lk] broadcastable to [B,1,R,Lk]
                            attn_mask_sub = attention_mask
                        teacher_logits = teacher_logits + attn_mask_sub
                    # Predictor logits on the same rows: [B,H,R,Lk]
                    Q_pred_sub = q_imp[:, :, loss_row_idx, :]           # [B,H,R,dDash]
                    K_pred_full = k_imp                                 # [B,H,Lk,dDash]
                    Q_pred_flat = Q_pred_sub.reshape(
                        bsz * self.num_heads, R, self.dDash
                    )
                    K_pred_flat = K_pred_full.reshape(
                        bsz * self.num_heads, key_len, self.dDash
                    )
                    student_logits = torch.bmm(
                        Q_pred_flat,
                        K_pred_flat.transpose(-2, -1)
                    ) / math.sqrt(self.dDash)
                    student_logits = student_logits.view(bsz, self.num_heads, R, key_len)
    
                    if attn_mask_sub is not None:
                        student_logits = student_logits + attn_mask_sub
    
                    # --- Loss selection on [B,H,R,Lk] ---
                    if self.lookahead != 0:
                        raise NotImplementedError("Row-subsampled aux loss with lookahead>0 is not supported yet.")
    
                    if self.softmax_causal_loss_mse:
                        target_dist = F.softmax(teacher_logits, dim=-1)
                        pred_dist   = F.softmax(student_logits, dim=-1)
                        loss = self.mseloss(pred_dist, target_dist)             # [B,H,R,Lk]
                        self.msemagn_loss = 1024 * loss.mean(dim=(-1, -2)).mean()
    
                    elif self.softmax_causal_loss_ce:
                        target_dist = F.softmax(teacher_logits, dim=-1).detach()
                        pred_dist   = F.softmax(student_logits, dim=-1)
                        ce = -(target_dist * (pred_dist + 1e-9).log()).sum(dim=-1)  # [B,H,R]
                        self.msemagn_loss = 0.1 * ce.mean()
    
                    elif getattr(self, "pairwise_loss", False):
                        # Pairwise logistic ranking on [B,H,R,Lk]
                        teacher_probs = F.softmax(teacher_logits, dim=-1)       # [B,H,R,Lk]
                        B_eff, H_eff, R_eff, Lk_eff = teacher_probs.shape
    
                        if attn_mask_sub is not None:
                            valid_mask = (attn_mask_sub == 0).expand(B_eff, H_eff, R_eff, Lk_eff)
                        else:
                            valid_mask = torch.ones_like(teacher_probs, dtype=torch.bool)
    
                        valid_counts = valid_mask.reshape(-1, Lk_eff).sum(-1)
                        has_valid = (valid_counts > 0).reshape(B_eff, H_eff, R_eff)
    
                        topk_ratio = getattr(self, "pairwise_topk_ratio", 0.2)
                        K = max(1, int(topk_ratio * Lk_eff))
    
                        probs_for_top = teacher_probs.masked_fill(~valid_mask, float("-inf"))
                        _, top_idx = probs_for_top.topk(K, dim=-1)              # [B,H,R,K]
    
                        probs_for_bot = teacher_probs.masked_fill(~valid_mask, 1.0)
                        _, bot_idx = probs_for_bot.topk(K, dim=-1, largest=False)  # [B,H,R,K]
    
                        student_logits_pw = student_logits.clamp(min=-1e4, max=1e4)
                        s = student_logits_pw                                    # [B,H,R,Lk]
                        s_pos = s.gather(-1, top_idx)                            # [B,H,R,K]
                        s_neg = s.gather(-1, bot_idx)                            # [B,H,R,K]
    
                        margin = s_pos - s_neg                                   # [B,H,R,K]
                        pairwise = F.softplus(-margin)
    
                        if attn_mask_sub is not None:
                            pairwise = pairwise * has_valid.unsqueeze(-1).to(pairwise.dtype)
    
                        self.msemagn_loss = pairwise.mean()
    
                    elif getattr(self, "pairwise_ce_loss", False):
                        teacher_probs = F.softmax(teacher_logits, dim=-1)       # [B,H,R,Lk]
                        pred_probs    = F.softmax(student_logits, dim=-1)       # [B,H,R,Lk]
                        B_eff, H_eff, R_eff, Lk_eff = teacher_probs.shape
    
                        if attn_mask_sub is not None:
                            valid_mask = (attn_mask_sub == 0).expand(B_eff, H_eff, R_eff, Lk_eff)
                        else:
                            valid_mask = torch.ones_like(teacher_probs, dtype=torch.bool)
    
                        valid_counts = valid_mask.reshape(-1, Lk_eff).sum(-1)
                        has_valid = (valid_counts > 0).reshape(B_eff, H_eff, R_eff)
    
                        topk_ratio = getattr(self, "pairwise_topk_ratio", 0.2)
                        K = max(1, int(topk_ratio * Lk_eff))
    
                        probs_for_top = teacher_probs.masked_fill(~valid_mask, float("-inf"))
                        _, top_idx = probs_for_top.topk(K, dim=-1)              # [B,H,R,K]
    
                        probs_for_bot = teacher_probs.masked_fill(~valid_mask, 1.0)
                        _, bot_idx = probs_for_bot.topk(K, dim=-1, largest=False)  # [B,H,R,K]
    
                        t_pos = teacher_probs.gather(-1, top_idx)               # [B,H,R,K]
                        t_neg = teacher_probs.gather(-1, bot_idx)
                        p_pos = pred_probs.gather(-1, top_idx)
                        p_neg = pred_probs.gather(-1, bot_idx)
    
                        t_pair = torch.stack([t_pos, t_neg], dim=-1)
                        p_pair = torch.stack([p_pos, p_neg], dim=-1)
    
                        t_pair = t_pair / (t_pair.sum(dim=-1, keepdim=True) + 1e-9)
                        p_pair = p_pair / (p_pair.sum(dim=-1, keepdim=True) + 1e-9)
    
                        pair_ce = -(t_pair * (p_pair + 1e-9).log()).sum(dim=-1)  # [B,H,R,K]
    
                        if attn_mask_sub is not None:
                            pair_ce = pair_ce * has_valid.unsqueeze(-1).to(pair_ce.dtype)
    
                        self.msemagn_loss = pair_ce.mean()
    
                    else:
                        raise ValueError("No loss selected for token importance predictor!")
    
                    # ---- Metrics (optional) on sub-sampled rows ----
                    if self.calc_hitrates:
                        est   = F.softmax(student_logits, dim=-1)
                        truth = F.softmax(teacher_logits, dim=-1)
                        # Standard top-50% token hit accuracy.
                        self.tok_hit_acc, self.tok_mean_rank_corr, self.tok_max_rank_corr = calculate_hit_metrics(
                            estimated_importance=est,
                            true_importance=truth,
                            top_k_ratio=0.5,
                        )
                        # Hard-token stats (top 10% / 5% teacher tokens).
                        self.tok_hit_acc_90, _, _ = calculate_hit_metrics(
                            estimated_importance=est,
                            true_importance=truth,
                            top_k_ratio=0.1,
                        )
                        self.tok_hit_acc_95, _, _ = calculate_hit_metrics(
                            estimated_importance=est,
                            true_importance=truth,
                            top_k_ratio=0.05,
                        )
        if self.layer_idx > 0 and self.train_headpredictor:
            head_importance_tensor = self.producer.head_importances[:, :, :, self.layer_idx % self.producer_frequency].float().to(attn_output.device)
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

                # Candidate tokens for *TokenButler* sparsity:
                #   - would have been active without TokenButler
                #   - are not sink tokens
                #   - are not in the sliding‑window tail.
                candidate_mask = (~attention_mask.bool())
                if self.min_sparse_index is not None and self.min_sparse_index > 0:
                    clamp_idx = min(self.min_sparse_index, true_mask.size(-1))
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
                print(f"Layer {self.layer_idx}: Effective Sparsity:", effective_sparsity, "%\t Sequence Length:", q_len)
        if self.layer_idx == 0:
            if self.effective_sparsity is None:
                self.effective_sparsity = 0.0

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, -1, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
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
                # k_len = k_importance.size(-1)
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
        return attn_output, attn_weights
        
def convert_kvcache_experimental(model, config, producer_frequency: int):
    """
    Replace all LlamaAttention blocks with LlamaAttentionExperimental.

    New modules are placed EXACTLY on the same device/dtype as the original
    LlamaAttention they replace. This keeps us consistent with the
    `device_map` that was used when loading the model.
    """
    layer_idx_counter = 0
    group_roots: dict[int, LlamaAttentionExperimental] = {}

    def recurse(parent_module: nn.Module, prefix: str = ""):
        nonlocal layer_idx_counter

        # We must use list(...) because we'll mutate parent_module._modules
        for name, child in list(parent_module._modules.items()):
            full_name = f"{prefix}.{name}" if prefix else name

            # Recurse into children first
            if len(list(child.children())) > 0:
                recurse(child, full_name)
            if isinstance(child, LlamaAttention):
                # Put the new attention on the *same device/dtype*
                # as the module we are replacing.
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

                new_attn = LlamaAttentionExperimental(
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