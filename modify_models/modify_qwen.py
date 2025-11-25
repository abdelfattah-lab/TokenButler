# qwen_attention_experimental.py

import math
import traceback
from typing import Optional, Tuple, Union, Dict, Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MSELoss

from transformers.cache_utils import Cache, DynamicCache

# Qwen2 / Qwen2.5 modeling bits
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    apply_rotary_pos_emb,
    repeat_kv,
)

# Your existing utilities / predictor stack
from utils import (
    calculate_hit_metrics,
    threshold_to_mask,
)
from predictor import (
    TokenImportancePredictorAttentive,
    PredictorDynamicCache,
    HeadImportancePredictor,
)


class QwenAttentionExperimental(nn.Module):
    """
    Drop-in replacement for Qwen2Attention that adds TokenButler-style
    token/head sparsity and predictor training hooks.

    It mirrors the structure of your LlamaAttentionExperimental class,
    but plugs into Qwen’s RoPE + mask + cache API.
    """

    def __init__(
        self,
        config,
        producer: Optional["QwenAttentionExperimental"] = None,
        layer_idx: int = 0,
        producer_frequency: int = 1,
        is_predictor_owner: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Qwen-specific metadata
        self.layer_type = (
            config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        )

        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.num_heads
        )
        self.max_position_embeddings = getattr(
            config, "max_position_embeddings", None
        )

        # Sliding-window layers (Qwen has alternating sliding layers)
        self.sliding_window = (
            getattr(config, "sliding_window", None)
            if self.layer_type == "sliding_attention"
            else None
        )

        # Inference / decode-time mode flag (set externally)
        self.inference_mode = False

        self.producer_frequency = max(1, int(producer_frequency))
        # Producer is the root layer that owns predictor outputs
        self.producer: Optional["QwenAttentionExperimental"] = producer

        # --- predictor / sparsity configuration ---------------------------------
        self.token_sparse_method: Optional[str] = None
        self.sparse_aggression: Optional[float] = None
        self.stream_llm_start_size = None
        self.dDash: Optional[int] = None
        self.intdim: Optional[int] = None
        self.attn_reduce_factor: Optional[int] = None
        self.head_attn_reduce_factor: Optional[int] = None
        self.effective_sparsity: Optional[float] = None
        self.min_sparse_index: Optional[int] = None

        self.pred_hid_size = self.hidden_size
        # Each predictor call produces one query tensor per consumer layer in
        # the group, so we set num_layers_pred = producer_frequency.
        self.num_layers_pred = self.producer_frequency
        self.num_tok_per_page = None
        self.calc_hitrates: bool = False
        self.flash_attn: bool = False

        # --- long-context aux loss knobs ----------------------------------------
        # Maximum number of query positions (rows) used for the aux loss.
        # Set to None or 0 to fall back to full L^2 (old behavior).
        self.max_loss_rows: Optional[int] = 256
        # Fraction of the sequence tail to sample from (we always include the last token).
        self.loss_tail_fraction: float = 0.5

        self.train_headpredictor: bool = False
        self.calibrate_thresholds: bool = False
        self.test_with_thresholds: bool = False
        self.late_context_upweight: bool = False
        self.softmax_causal_loss_mse: bool = False
        self.softmax_causal_loss_ce: bool = False
        self.old_predictor = None
        self.pairwise_loss: bool = False
        self.pairwise_ce_loss: bool = False
        self.mode: str = "balanced"  # "extreme_recall" or "balanced"
        # fraction of keys used as pos/neg in pairwise loss
        self.pairwise_topk_ratio: float = 0.02
        self.tokenbutler_variant: str = "tokenbutler_project"

        # Whether this layer runs the predictor (producer/root for the group)
        # and owns the q/k/head importance tensors.
        self.is_predictor_owner: bool = is_predictor_owner

        # --- sparsity control knobs (decode‑time only) --------------------------
        # For fixed_xpc: target_sparsity is fraction of *candidate* tokens pruned (0‑1).
        # For fixed_ytok: target_keep_tokens is the number of *candidate* tokens kept.
        self.target_sparsity: Optional[float] = None
        self.target_keep_tokens: Optional[int] = None
        # Decode‑time gating: prefill + first generated token stay dense.
        self._dense_kv_cutoff: int = 0
        self.always_dense_decode_tokens: int = 1

        if self.mode == "extreme_recall":  # top4‑style calibration
            self.low_recall_first: Dict[str, Tuple[int, int]] = {}
        elif self.mode == "balanced":  # top50‑style calibration
            self.low_recall_first = {}
        else:
            raise ValueError(f"Unknown sparsity mode {self.mode}")

        if getattr(self.config, "_name_or_path", None) not in self.low_recall_first:
            self.lowrecall_tuples = []
        else:
            self.lowrecall_tuples = self.low_recall_first[self.config._name_or_path]

        # Aux‑loss accumulators exist on all layers so downstream logging /
        # checkpoint code can always access them without attribute checks.
        self.msemagn_loss: Optional[torch.Tensor] = None
        self.headmsemagn_loss: Optional[torch.Tensor] = None

        if self.layer_idx > 0:
            self.mseloss = MSELoss(reduction="none")
            self.headmseloss = MSELoss(reduction="none")

        # Metrics / diagnostics (mirrors LlamaAttentionExperimental defaults)
        # and avoids AttributeError when training code probes them blindly.
        self.tok_hit_acc: float = 0.0
        self.tok_mean_rank_corr: float = 0.0
        self.tok_max_rank_corr: float = 0.0
        self.tok_hit_acc_90: float = 0.0
        self.tok_hit_acc_95: float = 0.0

        self.head_hit_acc: float = 0.0
        self.head_mean_rank_corr: float = 0.0
        self.head_max_rank_corr: float = 0.0

        self.true_threshmean: Optional[torch.Tensor] = None
        self.threshmean: Optional[torch.Tensor] = None
        self.final_mask_investigate: Optional[torch.Tensor] = None

        # Populated lazily by set_token_sparsity() when threshold calibration
        # is enabled; defined here to keep access sites simple.
        self.tok_calibration_set = None

        if self.is_predictor_owner:
            # This module stores the predictor outputs for its consumer group.
            # q_importance: [BH, producer_frequency, L, dDash]
            self.q_importance: Optional[torch.Tensor] = None
            self.k_importance: Optional[torch.Tensor] = None
            self.head_importances: Optional[torch.Tensor] = None
            self.actmagn_masklist: Dict[Any, Any] = {}
            self.available_tokens: Dict[Any, Any] = {}

        # --- core attention projection layers (match Qwen2Attention layout) -----
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # Qwen-specific convenience (not strictly needed, but kept for parity)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.is_causal = True

    # -------------------------------------------------------------------------
    # Global head-wise keep ratios (same machinery as in your Llama code)
    # -------------------------------------------------------------------------
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
        if not bad_pairs:  # no calibration data
            self._global_head_keep = keep
            return

        keep_max, keep_min = 1.0, float(self.sparse_aggression)
        N = len(bad_pairs)
        for rank, (head_idx, layer_idx) in enumerate(bad_pairs):
            frac = rank / (N - 1 + 1e-5)  # 0 … 1
            keep[layer_idx, head_idx] = keep_max - frac * (keep_max - keep_min)

        # --- global renormalisation ---------------------------------
        total_heads = L * H
        scale = (self.sparse_aggression * total_heads) / keep.sum()
        keep *= scale
        keep.clamp_(max=1.0)

        self._global_head_keep = keep  # cache

    def build_head_keep_ratios(self):
        """
        Return the [num_heads] vector for *this* layer, using
        model‑global calibration.  Safe to call every forward(); the
        table is computed once and cached.
        """
        if not hasattr(self, "_global_head_keep"):
            self._compute_global_head_keep()
        # put on same device as module parameters
        return self._global_head_keep[self.layer_idx].to(
            next(self.parameters()).device
        )

    # -------------------------------------------------------------------------
    # Predictor wiring
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Sparsity configuration (same contract as your Llama code)
    # -------------------------------------------------------------------------
    def set_token_sparsity(self):
        assert (
            self.token_sparse_method is not None
        ), "Set token_sparse_method first!"
        method = self.token_sparse_method

        # Optional: load per‑head threshold calibration if available.
        if method is not None:
            try:
                mname = self.config._name_or_path.split("/")[-1]
                read_path = f"threshold_calibs/{mname}/{method}.pkl"
                threshold_model_dictionary = torch.load(read_path)
                self.tok_calibration_set = threshold_model_dictionary
            except Exception:
                # If calibration is missing, we just run uncalibrated.
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
                    self.target_sparsity = x / 100.0  # prune fraction on candidates
                    self.sparse_aggression = (
                        1.0 - self.target_sparsity
                    )  # keep fraction on candidates

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

                    # Approximate keep fraction for logging / metrics only.
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
                    self.sparse_aggression = keep_frac
                    self.head_keep = None
            else:
                raise ValueError(
                    f"Unknown fixed sparsity spec '{spec}' in token_sparse_method='{method}'"
                )
        else:
            raise ValueError(
                f"Unsupported token sparsity method '{method}'. "
                "Use 'fixed_xpc' (e.g. fixed_65pc) or 'fixed_ytok' (e.g. fixed_128tok)."
            )

    # -------------------------------------------------------------------------
    # Decode-time gating helpers
    # -------------------------------------------------------------------------
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

        if self._dense_kv_cutoff == 0:
            # We don't know the prefill yet; treat this as first decode step.
            self._dense_kv_cutoff = kv_seq_len
            return False

        # Start pruning once kv_seq_len surpasses our dense cutoff.
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
        attn_valid = attention_mask[:, :, -1:, :] == 0  # [B,1,1,K]
        candidate_mask = attn_valid.expand(bsz, num_heads, 1, key_len)

        if min_sparse_index is not None and min_sparse_index > 0:
            clamp_idx = min(min_sparse_index, key_len)
            candidate_mask[..., :clamp_idx] = False

        if self.sliding_window is not None and self.sliding_window > 0:
            win = min(self.sliding_window, key_len)
            # Always keep the last 'sliding_window' keys.
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
            head_keep = head_keep.clamp(min=0.0, max=1.0).view(
                1, self.num_heads, 1, 1
            )
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
            raise ValueError(
                f"token_sparse_method '{method}' is not a fixed_* scheme"
            )

        # Rank candidate tokens by importance (descending), ignoring non‑candidates.
        scores = importance_scores.clone()
        scores = scores.masked_fill(~candidate_mask, float("-inf"))
        _, sorted_idx = scores.sort(dim=-1, descending=True)  # [B,H,1,K]

        B, H, _, K = sorted_idx.shape
        # rank[b, h, q, j] holds the rank position of key j for that (b,h,q)
        rank = torch.empty_like(sorted_idx, dtype=torch.long)
        arange_K = torch.arange(K, device=sorted_idx.device, dtype=torch.long)
        arange_K = arange_K.view(1, 1, 1, K).expand_as(sorted_idx)
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

    # -------------------------------------------------------------------------
    # Misc helpers
    # -------------------------------------------------------------------------
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

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

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[
            Union[Cache, DynamicCache, PredictorDynamicCache]
        ] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        This matches the call pattern of Qwen2Attention, but with extra
        features for TokenButler sparsity and predictor training.
        """
        bsz, q_len, _ = hidden_states.size()

        attn_device = self.q_proj.weight.device
        if hidden_states.device != attn_device:
            hidden_states = hidden_states.to(attn_device)
        if attention_mask is not None and attention_mask.device != attn_device:
            attention_mask = attention_mask.to(attn_device)
        if position_ids is not None and position_ids.device != attn_device:
            position_ids = position_ids.to(attn_device)
        if cache_position is not None and cache_position.device != attn_device:
            cache_position = cache_position.to(attn_device)

        # Some callers may pass output_attentions via **kwargs (HF style)
        if "output_attentions" in kwargs:
            output_attentions = kwargs.pop("output_attentions")

        tp = getattr(self.config, "pretraining_tp", 1)

        # --- projections to Q/K/V ------------------------------------------------
        if tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i]) for i in range(tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i]) for i in range(tp)
            ]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # [B, H, L, D]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # --- RoPE: Qwen passes (cos, sin) from model-level rotary_emb ----------
        if position_embeddings is not None:
            cos, sin = position_embeddings
            cos = cos.to(query_states.device)
            sin = sin.to(query_states.device)
            # Match HF Qwen2Attention signature: use explicit position_ids so
            # packed decoding / cache_position semantics stay consistent.
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
        else:
            # If you ever call this without position_embeddings, that's a setup error.
            raise ValueError(
                "QwenAttentionExperimental expects `position_embeddings` "
                "from Qwen2RotaryEmbedding."
            )

        # --- cache update -------------------------------------------------------
        if use_cache and past_key_values is not None:
            # Qwen DynamicCache expects these RoPE kwargs
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        kv_seq_len = key_states.shape[-2]

        # New-sequence prefill detection: q_len > 1 and kv_seq_len == q_len
        if (
            self.inference_mode
            and use_cache
            and q_len > 1
            and kv_seq_len == q_len
        ):
            # Remember prefill length and keep the next `always_dense_decode_tokens`
            # decode steps dense.
            self._dense_kv_cutoff = kv_seq_len + self.always_dense_decode_tokens

        final_mask = None

        # Expand KV heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_len = key_states.size(2)
        bsz, _, q_len, _ = query_states.size()

        # Fallback causal mask if none supplied (should not happen in normal Qwen2)
        if attention_mask is None:
            causal_mask_2d = torch.ones(
                q_len,
                kv_seq_len,
                device=hidden_states.device,
                dtype=torch.bool,
            ).triu(diagonal=1)
            causal_mask_4d = causal_mask_2d.unsqueeze(0).expand(
                bsz, 1, q_len, kv_seq_len
            )
            attention_mask = torch.full_like(
                causal_mask_4d, 0, dtype=hidden_states.dtype
            )
            if q_len != 1:
                attention_mask = attention_mask.masked_fill(
                    causal_mask_4d, float("-inf")
                )

        # ----------------------------------------------------------------------
        # Inference path (decode‑time TokenButler sparsity)
        # ----------------------------------------------------------------------
        evalmode = getattr(self, "eval_llm_mode", None)

        if self.inference_mode:
            if evalmode != "ExpPred":
                raise ValueError(
                    f"Unknown eval_llm_mode '{evalmode}' for inference TokenButler. "
                    "Expected 'ExpPred'."
                )

            min_sparse_index = self.min_sparse_index
            with torch.no_grad():
                if self.layer_idx > 0 and q_len == 1:
                    # Get predictor query for this layer from the group root
                    slot_idx = self._get_group_slot_index()
                    q_importance_tensor = self.producer.q_importance[
                        :, slot_idx, :, :
                    ].float().to(query_states.device)  # [BH, Lq, dDash]

                    Bk, Hk, Lk, Dh = key_states.shape

                    # Layer-specific projection to importance space
                    proj_weight = self.producer.sparse_token_predictor.key_cache_proj[
                        self.layer_idx
                    ]  # [H, Dh, dDash]

                    key_for_proj = key_states.to(
                        proj_weight.device, dtype=proj_weight.dtype
                    )  # [B,H,Lk,Dh]
                    k_proj = torch.einsum(
                        "bhlk,hkd->bhld", key_for_proj, proj_weight
                    )  # [B,H,Lk,dDash]
                    if k_proj.device != key_states.device:
                        k_proj = k_proj.to(key_states.device)

                    k_importance_tensor = k_proj.reshape(Bk * Hk, Lk, self.dDash)

                    importance_mask = torch.bmm(
                        q_importance_tensor,
                        k_importance_tensor.transpose(-2, -1),
                    ) / math.sqrt(self.dDash)  # [BH, Lq, Lk]
                    importance_mask = importance_mask.view(
                        bsz, self.num_heads, q_len, key_len
                    )  # [B, H, Lq, Lk]

                    # Teacher logits (for metrics only, when requested)
                    attn_logits = torch.matmul(
                        query_states, key_states.transpose(-2, -1)
                    ) / math.sqrt(self.head_dim)

                    if self.calc_hitrates:
                        estimated = nn.functional.softmax(
                            importance_mask + attention_mask, dim=-1
                        )
                        truth = nn.functional.softmax(
                            attn_logits + attention_mask, dim=-1
                        )
                        # Standard token hit accuracy: overlap of teacher/predictor
                        # top-50% keys.
                        (
                            self.tok_hit_acc,
                            self.tok_mean_rank_corr,
                            self.tok_max_rank_corr,
                        ) = calculate_hit_metrics(
                            estimated_importance=estimated,
                            true_importance=truth,
                            top_k_ratio=0.5,
                        )
                        # Hard-token diagnostics: how well we recover teacher
                        # tokens above the 90th / 95th percentile (top 10% / 5%).
                        (
                            self.tok_hit_acc_90,
                            _,
                            _,
                        ) = calculate_hit_metrics(
                            estimated_importance=estimated,
                            true_importance=truth,
                            top_k_ratio=0.1,
                        )
                        (
                            self.tok_hit_acc_95,
                            _,
                            _,
                        ) = calculate_hit_metrics(
                            estimated_importance=estimated,
                            true_importance=truth,
                            top_k_ratio=0.05,
                        )

                    if self.calibrate_thresholds:
                        # Same threshold investigation logic as in Llama code
                        unadj_importance_mask = importance_mask.clone()
                        importance_mask = torch.softmax(
                            importance_mask + attention_mask, dim=-1
                        )
                        sorted_values, sorted_ix = torch.sort(
                            importance_mask, dim=-1
                        )
                        sorted_true_values, _ = torch.sort(
                            torch.gather(
                                unadj_importance_mask, dim=-1, index=sorted_ix
                            ),
                            dim=-1,
                        )
                        idx = int(
                            importance_mask.size(-1) * self.sparse_aggression
                        )
                        true_thresholds = sorted_true_values[:, :, :, idx]
                        thresholds = sorted_values[:, :, :, idx]
                        self.true_threshmean = true_thresholds
                        self.threshmean = thresholds

                    if self.test_with_thresholds:
                        unadj_importance_mask = importance_mask.clone()
                        perhead_thresholds = self.tok_calibration_set[
                            self.layer_idx - 1
                        ].to(
                            unadj_importance_mask.device
                        )  # 0 does not have calibration data.
                        mask_tensor = threshold_to_mask(
                            unadj_importance_mask,
                            perhead_thresholds,
                            min_sparse_index,
                            bsz,
                            q_len,
                            key_len,
                        )
                    else:
                        importance_scores = torch.softmax(
                            importance_mask + attention_mask, dim=-1
                        )

                        apply_sparse = (
                            self.token_sparse_method is not None
                            and self._should_apply_sparse_decode(
                                q_len, kv_seq_len
                            )
                        )

                        if not apply_sparse or q_len != 1:
                            # Either prefill / first decode step / multi-token step:
                            # no TokenButler pruning.
                            mask_tensor = torch.zeros_like(importance_scores)
                        else:
                            mask_tensor = self._build_decode_mask_fixed(
                                importance_scores,
                                attention_mask,
                                min_sparse_index,
                            )

                    final_mask = mask_tensor
                    self.final_mask_investigate = final_mask
                else:
                    # Layer 0 or non-single-token decode: dense attention only.
                    final_mask = None

            # Build the final additive mask actually used for attention:
            if attention_mask is not None:
                if final_mask is not None and q_len == 1:
                    full_mask = attention_mask + final_mask
                else:
                    full_mask = attention_mask
            else:
                full_mask = (
                    final_mask if (final_mask is not None and q_len == 1) else None
                )

            if full_mask is not None:
                full_mask = full_mask.to(dtype=query_states.dtype)
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

        # ----------------------------------------------------------------------
        # Training / dense mode path
        # ----------------------------------------------------------------------
        else:
            if self.flash_attn:
                raise NotImplementedError(
                    "Flash attention with TokenButler is not implemented yet."
                )
            else:
                if attention_mask is not None:
                    attention_mask = attention_mask.to(dtype=query_states.dtype)
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

                attn_weights = None
                if output_attentions:
                    attn_weights = torch.matmul(
                        query_states, key_states.transpose(-2, -1)
                    ) / math.sqrt(self.head_dim)
                    if attention_mask is not None:
                        attn_weights = attn_weights + attention_mask
                    attn_weights = nn.functional.softmax(
                        attn_weights, dim=-1, dtype=torch.float32
                    ).to(value_states.dtype)

                # Mirror LlamaAttentionExperimental behaviour but be slightly
                # more defensive so we don't crash if the producer's predictor
                # hasn't been wired up yet.
                if self.layer_idx > 0:
                    # Predictor Q for this *group*: [BH, G, L, dDash]
                    slot_idx = self._get_group_slot_index()
                    producer = self.producer
                    if (
                        slot_idx is None
                        or producer is None
                        or getattr(producer, "q_importance", None) is None
                        or not hasattr(producer, "sparse_token_predictor")
                    ):
                        # Skip aux loss cleanly for this step; main model
                        # forward still runs dense attention.
                        self.msemagn_loss = torch.tensor(
                            0.0, device=query_states.device
                        )
                    else:
                        q_importance_tensor = producer.q_importance[
                            :, slot_idx, :, :
                        ].float().to(query_states.device)
                        BH, Lq_imp, _ = q_importance_tensor.shape
                        assert Lq_imp == q_len
                        assert BH == bsz * self.num_heads

                        # Reshape predictor Q to [B, H, Lq, dDash]
                        q_imp = q_importance_tensor.view(
                            bsz, self.num_heads, q_len, self.dDash
                        )

                        Bk, Hk, Lk, Dh = key_states.shape

                        # Use the projection corresponding to the *actual* layer index,
                        # NOT the predictor slot.
                        proj_weight = producer.sparse_token_predictor.key_cache_proj[
                            self.layer_idx
                        ]  # [H, Dh, dDash]
                        key_for_proj = key_states.to(
                            proj_weight.device, dtype=proj_weight.dtype
                        )
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

                        # --- derive per-example real lengths from attention_mask ----
                        if (
                            attention_mask is not None
                            and attention_mask.size(-2) == q_len
                        ):
                            # attention_mask: [B, 1, Lq, Lk], values 0 (keep) or -inf (mask)
                            row_valid = (
                                attention_mask[:, 0, :, :] > -1e4
                            ).any(dim=-1)  # [B, Lq] bool
                            real_lengths = row_valid.sum(
                                dim=-1
                            )  # [B], number of valid query tokens per example
                        else:
                            real_lengths = torch.full(
                                (bsz,),
                                q_len,
                                device=query_states.device,
                                dtype=torch.long,
                            )

                        global_max_valid = int(real_lengths.min().item())
                        if global_max_valid <= 0:
                            self.msemagn_loss = torch.tensor(
                                0.0, device=query_states.device
                            )
                        else:
                            effective_q_len = global_max_valid

                            # --- choose query rows for the loss -------------------
                            if (
                                getattr(self, "max_loss_rows", None) is None
                                or self.max_loss_rows <= 0
                                or self.max_loss_rows >= effective_q_len
                            ):
                                loss_row_idx = torch.arange(
                                    effective_q_len, device=query_states.device
                                )
                            else:
                                device = query_states.device
                                tail_frac = getattr(
                                    self, "loss_tail_fraction", 0.5
                                )
                                tail_len = max(
                                    1, int(effective_q_len * tail_frac)
                                )
                                tail_start = max(
                                    0, effective_q_len - tail_len
                                )
                                tail_range = torch.arange(
                                    tail_start,
                                    effective_q_len - 1,
                                    device=device,
                                )  # exclude last

                                num_extra = max(
                                    0, self.max_loss_rows - 1
                                )
                                if num_extra > 0 and tail_range.numel() > 0:
                                    rand_idx = tail_range[
                                        torch.randint(
                                            0,
                                            tail_range.numel(),
                                            (num_extra,),
                                            device=device,
                                        )
                                    ]
                                    last_idx = torch.tensor(
                                        [effective_q_len - 1],
                                        device=device,
                                    )
                                    loss_row_idx = torch.cat(
                                        [rand_idx, last_idx]
                                    )
                                else:
                                    loss_row_idx = torch.tensor(
                                        [effective_q_len - 1],
                                        device=device,
                                    )

                                loss_row_idx = torch.unique(loss_row_idx)

                            R = loss_row_idx.numel()

                            # Teacher logits on sub-sampled rows: [B,H,R,Lk]
                            Q_teacher_sub = query_states[
                                :, :, loss_row_idx, :
                            ]  # [B,H,R,D]
                            K_teacher_full = key_states  # [B,H,Lk,D]
                            teacher_logits = torch.matmul(
                                Q_teacher_sub,
                                K_teacher_full.transpose(-2, -1),
                            ) / math.sqrt(self.head_dim)

                            attn_mask_sub = None
                            if attention_mask is not None:
                                if attention_mask.size(-2) == q_len:
                                    attn_mask_sub = attention_mask[
                                        :, :, loss_row_idx, :
                                    ]
                                else:
                                    attn_mask_sub = attention_mask
                                teacher_logits = teacher_logits + attn_mask_sub

                            # Predictor logits on same rows: [B,H,R,Lk]
                            Q_pred_sub = q_imp[:, :, loss_row_idx, :]
                            K_pred_full = k_imp
                            Q_pred_flat = Q_pred_sub.reshape(
                                bsz * self.num_heads, R, self.dDash
                            )
                            K_pred_flat = K_pred_full.reshape(
                                bsz * self.num_heads, key_len, self.dDash
                            )

                            student_logits = torch.bmm(
                                Q_pred_flat,
                                K_pred_flat.transpose(-2, -1),
                            ) / math.sqrt(self.dDash)
                            student_logits = student_logits.view(
                                bsz, self.num_heads, R, key_len
                            )

                            if attn_mask_sub is not None:
                                student_logits = student_logits + attn_mask_sub

                            if self.softmax_causal_loss_mse:
                                target_dist = F.softmax(
                                    teacher_logits, dim=-1
                                )
                                pred_dist = F.softmax(
                                    student_logits, dim=-1
                                )
                                loss = self.mseloss(
                                    pred_dist, target_dist
                                )  # [B,H,R,Lk]
                                self.msemagn_loss = (
                                    1024 * loss.mean(dim=(-1, -2)).mean()
                                )

                            elif self.softmax_causal_loss_ce:
                                target_dist = F.softmax(
                                    teacher_logits, dim=-1
                                ).detach()
                                pred_dist = F.softmax(
                                    student_logits, dim=-1
                                )
                                ce = -(
                                    target_dist
                                    * (pred_dist + 1e-9).log()
                                ).sum(dim=-1)  # [B,H,R]
                                self.msemagn_loss = 0.1 * ce.mean()

                            elif getattr(self, "pairwise_loss", False):
                                teacher_probs = F.softmax(
                                    teacher_logits, dim=-1
                                )
                                (
                                    B_eff,
                                    H_eff,
                                    R_eff,
                                    Lk_eff,
                                ) = teacher_probs.shape

                                if attn_mask_sub is not None:
                                    valid_mask = (
                                        attn_mask_sub == 0
                                    ).expand(
                                        B_eff, H_eff, R_eff, Lk_eff
                                    )
                                else:
                                    valid_mask = torch.ones_like(
                                        teacher_probs, dtype=torch.bool
                                    )

                                valid_counts = valid_mask.reshape(
                                    -1, Lk_eff
                                ).sum(-1)
                                has_valid = (valid_counts > 0).reshape(
                                    B_eff, H_eff, R_eff
                                )

                                topk_ratio = getattr(
                                    self, "pairwise_topk_ratio", 0.2
                                )
                                K = max(
                                    1, int(topk_ratio * Lk_eff)
                                )

                                probs_for_top = teacher_probs.masked_fill(
                                    ~valid_mask, float("-inf")
                                )
                                _, top_idx = probs_for_top.topk(
                                    K, dim=-1
                                )

                                probs_for_bot = teacher_probs.masked_fill(
                                    ~valid_mask, 1.0
                                )
                                _, bot_idx = probs_for_bot.topk(
                                    K, dim=-1, largest=False
                                )

                                student_logits_pw = student_logits.clamp(
                                    min=-1e4, max=1e4
                                )
                                s = student_logits_pw
                                s_pos = s.gather(
                                    -1, top_idx
                                )  # [B,H,R,K]
                                s_neg = s.gather(
                                    -1, bot_idx
                                )

                                margin = s_pos - s_neg
                                pairwise = F.softplus(-margin)

                                if attn_mask_sub is not None:
                                    pairwise = pairwise * has_valid.unsqueeze(
                                        -1
                                    ).to(pairwise.dtype)

                                self.msemagn_loss = pairwise.mean()

                            elif getattr(self, "pairwise_ce_loss", False):
                                teacher_probs = F.softmax(
                                    teacher_logits, dim=-1
                                )
                                pred_probs = F.softmax(
                                    student_logits, dim=-1
                                )
                                (
                                    B_eff,
                                    H_eff,
                                    R_eff,
                                    Lk_eff,
                                ) = teacher_probs.shape

                                if attn_mask_sub is not None:
                                    valid_mask = (
                                        attn_mask_sub == 0
                                    ).expand(
                                        B_eff, H_eff, R_eff, Lk_eff
                                    )
                                else:
                                    valid_mask = torch.ones_like(
                                        teacher_probs, dtype=torch.bool
                                    )

                                valid_counts = valid_mask.reshape(
                                    -1, Lk_eff
                                ).sum(-1)
                                has_valid = (valid_counts > 0).reshape(
                                    B_eff, H_eff, R_eff
                                )

                                topk_ratio = getattr(
                                    self, "pairwise_topk_ratio", 0.2
                                )
                                K = max(
                                    1, int(topk_ratio * Lk_eff)
                                )

                                probs_for_top = teacher_probs.masked_fill(
                                    ~valid_mask, float("-inf")
                                )
                                _, top_idx = probs_for_top.topk(
                                    K, dim=-1
                                )

                                probs_for_bot = teacher_probs.masked_fill(
                                    ~valid_mask, 1.0
                                )
                                _, bot_idx = probs_for_bot.topk(
                                    K, dim=-1, largest=False
                                )

                                t_pos = teacher_probs.gather(
                                    -1, top_idx
                                )  # [B,H,R,K]
                                t_neg = teacher_probs.gather(
                                    -1, bot_idx
                                )
                                p_pos = pred_probs.gather(-1, top_idx)
                                p_neg = pred_probs.gather(-1, bot_idx)

                                t_pair = torch.stack(
                                    [t_pos, t_neg], dim=-1
                                )
                                p_pair = torch.stack(
                                    [p_pos, p_neg], dim=-1
                                )

                                t_pair = t_pair / (
                                    t_pair.sum(dim=-1, keepdim=True) + 1e-9
                                )
                                p_pair = p_pair / (
                                    p_pair.sum(dim=-1, keepdim=True) + 1e-9
                                )

                                pair_ce = -(
                                    t_pair * (p_pair + 1e-9).log()
                                ).sum(dim=-1)

                                if attn_mask_sub is not None:
                                    pair_ce = pair_ce * has_valid.unsqueeze(
                                        -1
                                    ).to(pair_ce.dtype)

                                self.msemagn_loss = pair_ce.mean()

                            else:
                                raise ValueError(
                                    "No loss selected for token importance predictor!"
                                )

                            if self.calc_hitrates:
                                est = F.softmax(student_logits, dim=-1)
                                truth = F.softmax(teacher_logits, dim=-1)
                                (
                                    self.tok_hit_acc,
                                    self.tok_mean_rank_corr,
                                    self.tok_max_rank_corr,
                                ) = calculate_hit_metrics(
                                    estimated_importance=est,
                                    true_importance=truth,
                                    top_k_ratio=0.5,
                                )
                                (
                                    self.tok_hit_acc_90,
                                    _,
                                    _,
                                ) = calculate_hit_metrics(
                                    estimated_importance=est,
                                    true_importance=truth,
                                    top_k_ratio=0.1,
                                )
                                (
                                    self.tok_hit_acc_95,
                                    _,
                                    _,
                                ) = calculate_hit_metrics(
                                    estimated_importance=est,
                                    true_importance=truth,
                                    top_k_ratio=0.05,
                                )

        # ----------------------------------------------------------------------
        # Head predictor (optional)
        # ----------------------------------------------------------------------
        if self.layer_idx > 0 and self.train_headpredictor:
            head_importance_tensor = self.producer.head_importances[
                :, :, :, self.layer_idx % self.producer_frequency
            ].float().to(attn_output.device)
            # attn_output: [B, H, L, D] after SDPA; we then transpose back later
            attn_head_weights = attn_output.mean(dim=-1).permute(
                0, 2, 1
            )  # [B, L, H]
            self.headmsemagn_loss = self.headmseloss(
                attn_head_weights, head_importance_tensor
            ).mean()

            if self.calc_hitrates:
                (
                    self.head_hit_acc,
                    self.head_mean_rank_corr,
                    self.head_max_rank_corr,
                ) = calculate_hit_metrics(
                    estimated_importance=head_importance_tensor,
                    true_importance=attn_head_weights,
                    top_k_ratio=0.5,
                )
        else:
            self.headmsemagn_loss = 0
            if self.calc_hitrates:
                self.head_hit_acc, self.head_mean_rank_corr, self.head_max_rank_corr = (
                    0,
                    0,
                    0,
                )

        # ----------------------------------------------------------------------
        # Effective sparsity measurement (decode-time only)
        # ----------------------------------------------------------------------
        checkeverytime = hasattr(self, "test_with_thresholds") and getattr(
            self, "test_with_thresholds"
        )
        if (
            getattr(self, "final_mask_investigate", None) is not None
            and attention_mask is not None
            and q_len == 1
        ):
            final_mask = self.final_mask_investigate
            if self.effective_sparsity is None or checkeverytime:
                true_mask = final_mask + attention_mask  # {0, -inf}

                candidate_mask = ~attention_mask.bool()
                if self.min_sparse_index is not None and self.min_sparse_index > 0:
                    clamp_idx = min(self.min_sparse_index, true_mask.size(-1))
                    candidate_mask[..., :clamp_idx] = False
                if self.sliding_window is not None and self.sliding_window > 0:
                    win = min(self.sliding_window, true_mask.size(-1))
                    candidate_mask[..., -win:] = False

                if candidate_mask.any():
                    total_deact = (true_mask.bool() & candidate_mask).sum(dim=-1)
                    causal_deact = (
                        attention_mask.bool() & candidate_mask
                    ).sum(dim=-1)
                    additional_deact = total_deact - causal_deact
                    num_candidates = candidate_mask.sum(dim=-1)
                    effective_sparsity = (
                        100
                        * (additional_deact.float() / num_candidates.float())
                        .mean()
                        .item()
                    )
                else:
                    effective_sparsity = 0.0
                self.effective_sparsity = effective_sparsity
                print(
                    f"Layer {self.layer_idx}: Effective Sparsity: {effective_sparsity:.2f}%\t Sequence Length: {q_len}"
                )
        if self.layer_idx == 0 and self.effective_sparsity is None:
            self.effective_sparsity = 0.0

        # ----------------------------------------------------------------------
        # Final projection + predictor bookkeeping
        # ----------------------------------------------------------------------
        # attn_output currently [B, H, L, D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, -1, self.hidden_size)

        if tp > 1:
            attn_output_chunks = attn_output.split(
                self.hidden_size // tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output_chunks[i], o_proj_slices[i])
                    for i in range(tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if self.is_predictor_owner and hasattr(self, "sparse_token_predictor"):
            try:
                q_importance, k_importance = self.sparse_token_predictor(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,  # same cache object
                    use_cache=use_cache,
                    layer_idx=self.layer_idx,
                )
                if self.train_headpredictor:
                    head_importances, _ = self.sparse_head_predictor(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        use_cache=use_cache,
                    )
                    head_importances = head_importances.view(
                        bsz, q_len, self.num_heads, self.num_hidden_layers
                    )  # [B, L, H, N]

            except Exception:
                print(traceback.format_exc())
                raise

            self.q_importance = q_importance
            self.k_importance = k_importance

            if self.train_headpredictor:
                if self.head_importances is None:
                    self.head_importances = head_importances
                else:
                    self.head_importances = torch.cat(
                        [self.head_importances, head_importances], dim=1
                    )

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Conversion helper
# ---------------------------------------------------------------------------

def convert_kvcache_experimental(
    model: nn.Module,
    config,
    producer_frequency: int,
) -> nn.Module:
    """
    Recursively walks `model` and replaces every Qwen2Attention module
    with QwenAttentionExperimental, wiring predictor group roots
    exactly like your Llama `convert_kvcache_experimental`.

    Args:
        model: A Qwen2/Qwen2.5 model instance (e.g. Qwen2ForCausalLM.model).
        config: Its config (e.g. model.config).
        producer_frequency: G = number of consumer layers served by one
            predictor "producer" layer (group size).
    """
    layer_idx_counter = 0
    group_roots: Dict[int, QwenAttentionExperimental] = {}

    def recurse(parent_module: nn.Module, prefix: str = ""):
        nonlocal layer_idx_counter

        # Use list(...) because we'll mutate parent_module._modules
        for name, child in list(parent_module._modules.items()):
            full_name = f"{prefix}.{name}" if prefix else name

            # Recurse into children first
            if len(list(child.children())) > 0:
                recurse(child, full_name)

            if isinstance(child, Qwen2Attention):
                # Put the new attention on the same device/dtype
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

                new_attn = QwenAttentionExperimental(
                    config=config,
                    producer=producer,
                    layer_idx=layer_idx,
                    producer_frequency=producer_frequency,
                    is_predictor_owner=is_owner,
                ).to(device=target_device, dtype=orig_dtype)

                # Copy over q/k/v/o weights from the original attention module.
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
