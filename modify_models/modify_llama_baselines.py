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

from utils import LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding, repeat_kv, sorted_index_to_mask
from utils import snapkv_mask_only, SlidingWindowCache, enforce_sliding_window
from transformers.cache_utils import DynamicCache

class LlamaAttentionExperimental(nn.Module):
    def __init__(self, config: LlamaConfig, producer=None, layer_idx=0):
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
        self.producer = producer
        self.layer_idx = layer_idx
        self.token_sparse_method = None
        self.sparse_aggression = None
        self.pruneax = None
        self.init_token_importance = None
        self.predictor_type = None
        self.stream_llm_start_size = None
        self.phead_scale = None
        self.dDash = None
        self.intdim = None
        self.oproj = None
        self.ll_six = None
        self.olayer = None
        self.add_attn = None
        self.ilayer = None
        self.min_sparse_index = None
        self.no_pred_causal_mask = None
        self.effective_sparsity = None
        self.replace_attention = None
        self.post_proj_causal = None
        self.pred_hid_size = self.hidden_size
        self.num_tok_per_page = None
        self.actmagn_masklist = {}
        if self.layer_idx > 0:
            self.mseloss = MSELoss(reduction='none')
            self.msemagn_loss = None
            self.celoss = nn.CrossEntropyLoss(reduction='none')
            self.kldivloss = torch.nn.KLDivLoss(reduction='none')
        
        # Attention setup
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()
        
    def update_predictor(self):
        pass
    
    def generate_ll_six(self, q_len):
        ll_six = []
        for curr_l in range(1, q_len+1):
            mt = [0,1,2,3] + list(range(4, curr_l, 1))[::-1]
            remmt = list(set(list(range(q_len))) - set(mt))
            mt = mt + remmt
            ll_six.append(mt)
        self.ll_six = torch.tensor(ll_six).to(self.q_proj.weight.device)

    def set_head_sparsity(self, head_sparsity_aggression, global_prune):
        self.head_sparsity_aggression = head_sparsity_aggression
        self.head_global_prune = global_prune

    def set_token_sparsity(self):
        assert self.token_sparse_method is not None, "Set token sparse method first!"
        if self.token_sparse_method == "LazyLLM":
            if self.layer_idx <= 9:
                self.sparse_aggression = 1
            elif self.layer_idx <= 19:
                self.sparse_aggression = 0.7
            elif self.layer_idx <= 28:
                self.sparse_aggression = 0.4
            else:
                self.sparse_aggression = 0.1
        elif "fixed" in self.token_sparse_method:
            if self.layer_idx == 0:
                self.sparse_aggression = 1
            else:
                self.sparse_aggression = 1 - float(self.token_sparse_method.split("_")[1].split("pc")[0])/100.
        elif "progressive" in self.token_sparse_method:
            pc_drop = float(self.token_sparse_method.split("_")[1].split("pc")[0])/100.
            self.sparse_aggression = (1 - pc_drop) ** (self.layer_idx)  # (x% per layer, progressive_xpc style)
        else:
            raise ValueError(f"Unknown token sparsity method {self.token_sparse_method}")
            

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                config=self.config
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

    @torch.inference_mode()                        # no grad & small overhead
    def tactic_prefill(self, key_states: torch.Tensor):
        B, kvH, T, D = key_states.shape
        # paper default is m = 32; keep tensor for dtype/device parity
        self.cluster_sz = torch.tensor(32, device=key_states.device)
        C               = math.ceil(T / self.cluster_sz.item())     # clusters/head

        # (1)  reshape once, keep fp16 / bf16
        k = key_states.view(B * kvH, T, D)
        # (2)  k‑means++‑like init (farthest‑point along sequence stride)
        # init_idx = torch.linspace(0, T - 1, steps=C, dtype=torch.long,
        #                           device=k.device)
        # init     = k.gather(1, init_idx.unsqueeze(-1).expand(-1, -1, D))

        init_idx = torch.linspace(
            0, T - 1, steps=C, dtype=torch.long, device=k.device
        )                                           # (C,)
        # add the missing batch dimension and repeat for every (B·kvH) row
        init_idx = init_idx.unsqueeze(0).expand(k.size(0), -1)     # (B·kvH, C)

        # gather the vectors that sit at those positions
        init = k.gather(
            1,                                           # along sequence axis
            init_idx.unsqueeze(-1)                       # (B·kvH, C, 1)
                    .expand(-1, -1, D)                   # (B·kvH, C, D)
        )
        # (3)  two Lloyd iterations – empirically halves distortion
        dist  = (k.unsqueeze(2) - init.unsqueeze(1)).pow_(2).sum(-1)
        label = dist.argmin(-1)
        for _ in range(2):
            cent = torch.zeros_like(init)
            cnt  = torch.zeros_like(init[..., :1])
            cent.scatter_add_(1, label[..., None].expand_as(k), k)
            cnt.scatter_add_(1, label[..., None], torch.ones_like(k[..., :1]))
            cent = cent / cnt.clamp_min_(1)
            dist  = (k.unsqueeze(2) - cent.unsqueeze(1)).pow_(2).sum(-1)
            label = dist.argmin(-1)
        centroids = cent
        # (4)  centroid update via scatter_add → fused kernel
        centroids = torch.zeros_like(init)
        counts    = torch.zeros_like(init[..., :1])
        centroids.scatter_add_(1, label[..., None].expand_as(k), k)
        counts.scatter_add_(1, label[..., None], torch.ones_like(k[..., :1]))
        centroids = centroids / counts.clamp_min_(1)

        # (5)  store
        self.cents       = centroids.view(B, kvH, C, D)
        self.cluster_id  = label.view(B, kvH, T)
        self.intra_pos   = (torch.arange(T, device=k.device)
                            .expand(B*kvH, T) -            # trick: cumulative
                            torch.cumsum(label.roll(1, -1) != label, dim=1)
                        ).view(B, kvH, T)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        if q_len != 1: # this is prefill stage for first token output, reset self.token_mask
                       # further, this should guarantee that token_mask is always assigned, as its always prefill first.
            self.token_mask = None


        try:
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
        except Exception as e:
            import pdb; pdb.set_trace()
        

        evalmode = self.eval_llm_mode
        num_tokens_to_keep = int(q_len * self.sparse_aggression)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)  # AHMED: Modified this to use the newer version.
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if use_cache:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        kv_seq_len = key_states.shape[-2]

        final_mask = None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_len = key_states.size(2)
        bsz, q_len = query_states.size(0), query_states.size(2)


        if attention_mask is None:
            # We want a [q_len, kv_seq_len] boolean upper-triangular mask
            causal_mask_2d = torch.ones(q_len, kv_seq_len, 
                                        device=hidden_states.device, 
                                        dtype=torch.bool).triu(diagonal=1)
            # Then shape it to [bsz, 1, q_len, kv_seq_len]
            causal_mask_4d = causal_mask_2d.unsqueeze(0).expand(bsz, 1, q_len, kv_seq_len)
            # Now fill -inf where the mask is True
            attention_mask = torch.full_like(causal_mask_4d, 0, dtype=hidden_states.dtype)
            if q_len != 1:
                attention_mask = attention_mask.masked_fill(causal_mask_4d, float("-inf"))

        tactic_P = getattr(self, "tactic_threshold", 0.99)   # default 90 % mass
                
        # ---- MagicPig defaults -------------------------------------------
        self.MP_NUM_SINK   = 4
        self.MP_NUM_LOCAL  = 4
        self.MP_K          = 10          # hyperplanes per row
        self.MP_L          = 8         # number of rows
        self.MP_HASH_FUNC  = torch.randn(
                self.head_dim, self.MP_K * self.MP_L, dtype=torch.float16
        ).to(self.q_proj.weight.device)
        # will be broadcast to every head; you can seed() for reproducibility
        # cache for the running hash codes (filled during the loop)
        self._mp_hash_cache = None       # lazily allocated
        attn_o_precalc = False
        min_sparse_index = self.min_sparse_index
        with torch.no_grad():
            if evalmode == "dense":
                attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            elif evalmode == "tactic_sim":
                attn_weights = torch.matmul(
                    query_states, key_states.transpose(-2, -1)
                ) / math.sqrt(self.head_dim)
                groups  = self.num_key_value_groups
                self.tactic_prefill(key_states)
                # build / refresh centroids only when cache *grew*
                if (not hasattr(self, "cents")                or
                    self.cents is None                       or
                    self.cents.size(2) != math.ceil(kv_seq_len /
                                                    self.cluster_sz.item())):
                    self.tactic_prefill(key_states.detach())
                if self.layer_idx == 0:
                    # keep layer‑0 dense – many sparse papers do that
                    pass
                else:
                    P          = getattr(self, "tactic_threshold", 0.9)
                    # P = 0.8
                    cents      = self.cents            # [B,kvH,C,D]
                    cid        = self.cluster_id       # [B,kvH,T]
                    # cents      = self.producer.cents            # [B,kvH,C,D]
                    # cid        = self.producer.cluster_id       # [B,kvH,T]
                    kvH      = cents.size(1)               # 24
                    heads_q  = query_states.size(1)        # 24
                    groups_q = heads_q // kvH              # 1 (must divide)
                    cid_rep  = cid.repeat_interleave(groups_q, dim=1)   # (B,24,T)

                    pos_in_c   = self.intra_pos        # [B,kvH,T]
                    cluster_sz = int(self.cluster_sz.item()) 
                    # pos_in_c   = self.producer.intra_pos        # [B,kvH,T]
                    # cluster_sz = int(self.producer.cluster_sz.item()) 
                    C          = cents.size(2)
                    heads = heads_q   
                    kvH        = cents.size(1)
                    B, _, T, D = key_states.shape
                    device, dt = key_states.device, key_states.dtype
                    kept_k   = torch.empty(B, heads, 0, D, dtype=dt, device=device)
                    kept_v   = torch.empty_like(kept_k)
                    kept_idx = torch.empty(B, heads, 0, dtype=torch.long, device=device)
                    final_mask  = torch.full((B, heads, q_len, T),
                                            float('-inf'), device=device, dtype=dt)
                    attn_weights = final_mask.clone()
                    # ❷ A/x + b parameters (one per head) initialised from prefill
                    #    Fit on 2 % random sample of logits per head (Alg.1 in paper)
                    sample = torch.randperm(T, device=device)[: max(1, int(0.02*T))]
                    k_samp = key_states[:, :, sample, :]                  # [B,kvH,S,D]
                    kvH = cents.size(1)                     # already defined (= 24 here)
                    q0 = query_states[:, :kvH, 0:1, :]      # (B, kvH, 1, D)
                    q0 = q0.expand(-1, kvH, len(sample), -1)
                    logits_sample = (q0 * k_samp).sum(-1).float()         # [B,kvH,S]
                    logits_sample = logits_sample.sort(-1).values         # ascending
                    # logits_sample = logits_sample.sort(-1, descending=True).values
                    # simple reciprocal fit:  prob ≈ a/(rank)+b
                    ranks   = torch.arange(1, logits_sample.size(-1)+1,
                                        device=device).float()
                    prob    = torch.softmax(logits_sample, -1)
                    inv_r   = 1.0 / ranks
                    A       = ((prob - prob.mean(-1, keepdim=True)) * inv_r).sum(-1)
                    B_par   = prob.mean(-1) - A * inv_r.mean()            # [B,kvH]
                    # repeat to full num_heads
                    A     = A.repeat_interleave(groups_q, dim=1)      # (B, 72)
                    B_par = B_par.repeat_interleave(groups_q, dim=1)  # (B, 72)
                    heads = heads_q

                    H_R = torch.cumsum(1. / torch.arange(1, C + 1, device=device, dtype=dt), 0)  # (C,)
                    for t in range(q_len):
                        q_t = query_states[:, :, t:t+1]
                        cent_rep = cents.repeat_interleave(groups_q, dim=1)          # (B,H,C,D)
                        c_scores = torch.matmul(q_t, cent_rep.transpose(-2, -1)).squeeze(2)
                        c_idx = c_scores.argsort(-1, descending=True)         # (B,H,C)
                        rank_idx = torch.arange(1, C + 1, device=device, dtype=dt)  # (C,)
                        cum_mass = A.unsqueeze(-1)*H_R + B_par.unsqueeze(-1)*rank_idx
                        # smallest R s.t. cum_mass ≥ P
                        R_est = (cum_mass < P).sum(-1) + 1
                        R_est = R_est.clamp(min=4, max=C)
                        N_est = (R_est * self.cluster_sz).clamp_max_(t + 1)
                        # R_est = (rhs <= (1 - P)).sum(-1).clamp(min=1)
                        # N_est = (R_est * cluster_sz).clamp_max_(t + 1)        # (B,H)
                        row_mask = torch.full((B, heads, 1, T), float('-inf'), device=device, dtype=dt)
                        # ------------------------------------------------------------
                        # ❸ Build a per‑head map:  cluster_id  ->  rank (1 .. C)
                        # ------------------------------------------------------------

                        rank_map = torch.empty_like(c_idx)
                        rank_map.scatter_(2, c_idx,
                                          torch.arange(1, C + 1, device=device)
                                               .view(1, 1, -1)
                                               .expand_as(c_idx))
                        # Already (B, heads, C) – no further repeat needed
                        tok_rank  = rank_map.gather(-1, cid_rep)     
                        # ------------------------------------------------------------
                        # ❹ Tokens whose cluster rank  ≤  R_est —and— that are causal
                        # ------------------------------------------------------------
                        causal_mask  = torch.arange(T, device=device).unsqueeze(0).unsqueeze(0) <= t
                        gather_mask  = (tok_rank <= R_est.unsqueeze(-1)) & causal_mask          # (B,H,T)
                        rank_in_mask = torch.cumsum(gather_mask.to(torch.int32), dim=-1)   # (B,H,T)
                        # keep_cond = gather_mask & (rank_in_mask <= N_est.unsqueeze(-1))
                        # keep_cond = gather_mask
                        rank_in_mask = torch.cumsum(
                            gather_mask.to(torch.int32), dim=-1
                        )                                                    # (B, heads, T)
                        keep_cond = gather_mask & (
                            rank_in_mask <= N_est.unsqueeze(-1)
                        )          
                        row_mask.masked_fill_(keep_cond.unsqueeze(2), 0.0)
                        # logits = torch.matmul(q_t, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                        # logits[:, :, :, : t + 1] += row_mask[:, :, :, : t + 1]
                        logits = (q_t @ key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                        logits += row_mask
                        final_mask[:, :, t:t+1, :] = row_mask
                        attn_weights[:, :, t:t+1, :] = logits
                        

            elif evalmode == "magicpig":
                # ---------------------------------------------
                # 0. Dense dot products (you already had this)
                # ---------------------------------------------
                attn_weights = torch.matmul(
                    query_states, key_states.transpose(-2, -1)
                ) / math.sqrt(self.head_dim)

                if self.layer_idx == 0:
                    # keep layer‑0 dense – many sparse papers do that
                    pass
                else:
                    # ------------------------------------------------------------
                    # 1.  Build / extend the hash cache for *all* past keys
                    # ------------------------------------------------------------
                    bsz, n_heads, q_len, kv_seq_len = attn_weights.shape
                    kv_heads = self.num_key_value_heads
                    n_heads_k = key_states.size(1)   
                    device   = attn_weights.device

                    # allocate once:  [B, kvH, L, seq_len]  int16
                    if (self._mp_hash_cache is None
                        or self._mp_hash_cache.size(1) != n_heads_k
                        or self._mp_hash_cache.size(-1) < kv_seq_len):
                        self._mp_hash_cache = torch.empty(
                            bsz, n_heads_k, self.MP_L, kv_seq_len,
                            dtype=torch.int16, device=device
                        )

                    # hash every *new* key that appeared since last call
                    with torch.no_grad():
                        # keys are [B, kvH, seq_len, D]
                        new_slice = key_states.size(-2)    # kv_seq_len == key_len
                        k_flat = key_states               \
                                .permute(0,1,3,2)        \
                                .reshape(-1, self.head_dim, new_slice)
                        # sign(k  @  W)  →  {0,1}
                        h = torch.matmul(
                                k_flat.transpose(1,2).contiguous().view(-1, self.head_dim),
                                self.MP_HASH_FUNC
                            ).gt_(0).view(
                                -1, new_slice, self.MP_L, self.MP_K
                            ).to(torch.int16)
                        # pack K bits per row into a single int16
                        bits = (h * (1 << torch.arange(self.MP_K, device=device))).sum(-1)
                        bits = bits.permute(0,2,1).contiguous()          # [B·kvH, L, new]
                        bits = bits.view(bsz, n_heads_k, self.MP_L, new_slice) 
                        self._mp_hash_cache[..., :new_slice] = bits

                    # ------------------------------------------------------------
                    # 2.   Build the keep‑mask “as if decoding”
                    # ------------------------------------------------------------
                    comb   = bsz * n_heads
                    A      = attn_weights.view(comb, q_len, kv_seq_len)
                    keep   = torch.full_like(A, float('-inf'))

                    sink_idx = torch.arange(min(self.MP_NUM_SINK, kv_seq_len),
                                            device=device, dtype=torch.long)

                    for i in range(q_len):                        # simulate decode
                        # -------- mandatory ------------------------------------
                        keep[:, i, sink_idx] = 0.0
                        local_start = max(sink_idx[-1].item()+1, i - self.MP_NUM_LOCAL + 1)
                        if local_start <= i:
                            local_rng = torch.arange(local_start, i+1, device=device)
                            keep[:, i, local_rng] = 0.0

                        # -------- LSH retrieval --------------------------------
                        # hash for the *query* token i (shape [B, kvH, L])
                        qk = key_states[:, :, i]                  # [B, kvH, D]
                        qhash = (qk @ self.MP_HASH_FUNC)          \
                                    .gt_(0).to(torch.int16)       \
                                    .view(bsz, n_heads_k, self.MP_L, self.MP_K)
                        qhash = (qhash * (1 << torch.arange(self.MP_K, device=device))).sum(-1)

                        # compare row‑by‑row: broadcast equals
                        #    cache: [B, kvH, L, kv_seq_len]
                        #    qhash: [B, kvH, L, 1]
                        match = (self._mp_hash_cache[..., :kv_seq_len] ==
                                qhash.unsqueeze(-1))        # [B, n_heads_k, MP_L, kv_seq_len]
                        lsh_keep = match.any(dim=2)          # [B, n_heads_k, kv_seq_len]
                        # no need to repeat_kv – we are already at n_heads
                        lsh_keep = lsh_keep.view(comb, kv_seq_len)

                        keep[:, i].masked_fill_(lsh_keep, 0.0)

                    # final causal warm‑up etc.
                    keep[:, :, :self.min_sparse_index] = 0.0
                    # ------------------------------------------------------------
                    self.final_mask_investigate = keep.clone()  # save for debugging
                    final_mask = keep.clone()  # save for debugging
                    attn_weights = attn_weights + keep
            elif evalmode in ["oracle", "random", "init_oracle", "lookahead_oracle"] or "oracle" in evalmode:
                oracle_attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                oracle_attn_weights = oracle_attn_weights + attention_mask
                oracle_attn_weights = nn.functional.softmax(oracle_attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
                importance_mask = oracle_attn_weights.detach().float()
                importance_mask = torch.softmax(importance_mask, dim=-1, dtype=torch.float32)
                if evalmode == "random":
                    importance_mask = torch.softmax(torch.rand_like(importance_mask) + attention_mask, dim=-1, dtype=torch.float32)
                if evalmode in ["init_oracle", "lookahead_oracle"]:
                    if self.layer_idx > 0:
                        importance_mask = self.producer.init_token_importance
                if evalmode == "oracle":
                    save_importance_mask = importance_mask.detach().float() # [B, H, L, L]
                    save_importance_mask = save_importance_mask.permute(0, 2, 3, 1) # [B, L, L, H]
                else:
                    save_importance_mask = importance_mask
                if self.layer_idx > 0:
                    if self.sparse_aggression < 1:
                        if evalmode == "oracle":
                            _, sorted_indices = importance_mask.sort(dim=-1, descending=True)  # [B, H, q_len, key_len]
                        elif evalmode in ["init_oracle", "lookahead_oracle"]:
                            importance_mask = importance_mask.mean(dim=1, keepdim=True).expand_as(importance_mask)
                            _, sorted_indices = importance_mask.sort(dim=-1, descending=True)  # [B, H, q_len, key_len]
                        else:
                            _, sorted_indices = importance_mask.sort(dim=-1, descending=True)
                        

                        sorted_indices = sorted_indices[:, :, -q_len:, :]
                        mask_tensor = sorted_index_to_mask(sorted_indices, attention_mask, min_sparse_index, bsz, q_len, key_len, self.sparse_aggression, self.sliding_window)
                        if self.sliding_window is not None:
                            if not hasattr(self, "window_cache"):
                                self.window_cache = SlidingWindowCache(max_seq_len=1024,
                                                                    sliding_window=self.sliding_window,
                                                                    device=mask_tensor.device)
                            window = self.window_cache.get_window(q_len, key_len)
                            mask_tensor = enforce_sliding_window(mask_tensor, window)
                        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                        final_mask = mask_tensor
                        self.final_mask_investigate = final_mask
                        attn_wt_shape = attn_weights.shape
                        # if q_len == 1:
                        #     import pdb; pdb.set_trace()
                        if q_len != 1:
                            attn_weights = attn_weights + mask_tensor + attention_mask
                        else:
                            attn_weights = attn_weights + mask_tensor
                        if attn_weights.shape != attn_wt_shape:
                            import pdb; pdb.set_trace()
                        assert attn_weights.shape == attn_wt_shape, f"Shape mismatch {attn_weights.shape} {attn_wt_shape} due to MT {mask_tensor.shape}"
                    else:
                        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                else:
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            elif evalmode == "streamingLLM":
                if self.layer_idx > 0:
                    # if self.ll_six is None or self.ll_six.size(-1) != q_len:
                    self.generate_ll_six(key_len)
                    ll_six = self.ll_six
                    # here, it should be q_len, key_len i think. -- init max size and then pick
                    sorted_indices = ll_six.unsqueeze(0).unsqueeze(0).expand(bsz, self.num_heads, key_len, key_len).to(query_states.device)
                    sorted_indices = sorted_indices[:, :, -q_len:, :]
                    mask_tensor = sorted_index_to_mask(sorted_indices, attention_mask, min_sparse_index, bsz, q_len, key_len, self.sparse_aggression, None)
                    if self.sliding_window is not None:
                        if not hasattr(self, "window_cache"):
                            self.window_cache = SlidingWindowCache(max_seq_len=1024,
                                                                sliding_window=self.sliding_window,
                                                                device=mask_tensor.device)
                        window = self.window_cache.get_window(q_len, key_len)
                        mask_tensor = enforce_sliding_window(mask_tensor, window)
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    final_mask = mask_tensor
                    attn_weights = attn_weights + mask_tensor + attention_mask
                else:
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            elif evalmode == "snapkv":
                """
                Incremental SnapKV approach that mimics 'h2o_true':
                - We keep an active set of tokens of max size 'max_budget'.
                - Once a token is pruned, we never pick it again.
                - We use a SnapKV-like metric (aggregated attention from a local observation window)
                    to decide which tokens remain in the active set.
                """
                if not hasattr(self, "snapkv_cache"):
                    self.snapkv_cache = None
                    # 1) Standard scaled-dot product attention

                if self.layer_idx > 0:
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    bsz, num_heads, q_len, kv_seq_len = attn_weights.size()
                    self.snapkv_cache = None
                    combined_bh = bsz * num_heads                    
                    attn_weights_2d = attn_weights.view(combined_bh, q_len, kv_seq_len)

                    if not hasattr(self, "causal_mask") or self.causal_mask.shape[0] != q_len or self.causal_mask.shape[1] != kv_seq_len:
                        big_mask = torch.full((q_len, kv_seq_len), float('-inf'), device=attn_weights.device)
                        for row in range(q_len):
                            big_mask[row, :row+1] = 0.0
                        self.causal_mask = big_mask
                    else:
                        big_mask = self.causal_mask[:q_len, :kv_seq_len] 
                    attn_weights_2d = attn_weights_2d + big_mask.unsqueeze(0) 
                    attn_weights_2d = F.softmax(attn_weights_2d, dim=-1)      

                    # Prefix Sum: On query, convert to cumulative sums across queries
                    # Efficient way of keeping cumulative attention weights instead of recomputing per-window
                    # Line 11 : vote = attn_weights[..., -window_size:, :-window_size].sum(dim=-2)
                    prefix_sums_2d = attn_weights_2d.cumsum(dim=1)

                    final_mask = torch.full_like(attn_weights, float('-inf'))
                    final_mask_2d = final_mask.view(combined_bh, q_len, kv_seq_len)

                    max_budget = max(int(q_len * self.sparse_aggression), min_sparse_index)
                    # max_budget = min(1024, max(kv_seq_len, min_sparse_index))
                    # max_budget = 1024
                    active_tokens = torch.full((combined_bh, max_budget), 0, dtype=torch.long, device=attn_weights.device)
                    active_counts = torch.ones(combined_bh, dtype=torch.long, device=attn_weights.device)

                    final_mask_2d[:, 0, 0] = 0.0
                    # obs_size = 16
                    # obs_size = 4
                    if self.sliding_window is not None:
                        obs_size = self.sliding_window
                    else:
                        obs_size = 4
                    
                    for i in range(1, q_len):
                        # step_budget = max(int((i + 1 - obs_size) * self.sparse_aggression), min_sparse_index)
                        # Either no step budget, or a step budget that increases linearly with query index
                        # Sliding window and anchor is guaranteed, so we msut subtract from budget
                        step_budget = max(int((i + 1 - obs_size - min_sparse_index) * self.sparse_aggression), 1)
                        # step_budget = max(int((i + 1) * self.sparse_aggression), min_sparse_index)
                        # step_budget = max_budget
                        obs_start = max(0, i - obs_size + 1)
                        obs_length = i - obs_start + 1
                        prefix_length = obs_start

                        # Our prefix sum was 'cumulative' over ALL past queries. 
                        # We'll write this into a buffer "aggregator" that only keeps the prefix sum over the observation window.
                        aggregator = torch.zeros(combined_bh, i + 1, device=attn_weights.device)
                        if obs_start > 0:
                            # To keep only observation window, we need to 'remove' the prefix sum up to obs_start.
                            aggregator[:, : (i + 1)] = prefix_sums_2d[:, i, : (i + 1)] - prefix_sums_2d[:, obs_start - 1, : (i + 1)]
                        else:
                            aggregator[:, : (i + 1)] = prefix_sums_2d[:, i, : (i + 1)]

                        # Line 13: pool_vote = pool1d(vote, kernel_size = kernel_size , padding = kernel_size //2 , stride =1)
                        kernel_size = 5
                        aggregator_reshaped = aggregator[:, : (i + 1)].unsqueeze(1)
                        aggregator_pooled = F.max_pool1d(aggregator_reshaped, kernel_size=kernel_size,
                                                        stride=1, padding=kernel_size // 2)
                        aggregator_pooled = aggregator_pooled.squeeze(1)

                        new_token_importance = aggregator_pooled[:, i].unsqueeze(-1)

                        # We need to track active tokens and track budget for each B*H
                        can_add = active_counts < step_budget
                        add_indices = can_add.nonzero(as_tuple=False).squeeze(-1)
                        active_tokens[add_indices, active_counts[add_indices]] = i
                        active_counts[add_indices] += 1

                        cannot_add = ~can_add
                        try:
                            # If any heads have exceeded budget, we need to replace tokens
                            if cannot_add.any():
                                replace_indices = cannot_add.nonzero(as_tuple=False).squeeze(-1)
                                # get active tokens for budget excess
                                current_active = active_tokens[replace_indices, :step_budget]
                                # Get their pooled importances
                                row_imps = aggregator_pooled[replace_indices].gather(1, current_active)
                                # find least important token
                                min_vals, min_idxs = torch.min(row_imps, dim=1, keepdim=True)
                                # replace if new token is more important
                                new_imps = new_token_importance[replace_indices]
                                should_replace = new_imps > min_vals
                                rows_to_replace = replace_indices[should_replace.squeeze(1)]
                                pos_to_replace = min_idxs[should_replace.squeeze(1)].squeeze(1)
                                active_tokens[rows_to_replace, pos_to_replace] = i
                        except:
                            import pdb; pdb.set_trace()

                        # Initialize mask for that 'query index'
                        final_mask_2d[:, i, :] = float('-inf')
                        positions = torch.arange(max_budget, device=attn_weights.device).unsqueeze(0)
                        valid_positions = positions < active_counts.unsqueeze(1)
                        valid_rows = valid_positions.nonzero(as_tuple=True)[0]
                        valid_token_positions = valid_positions.nonzero(as_tuple=True)[1]
                        # Used 0,1 to get 'bh' and 'token' positions
                        valid_tokens = active_tokens[valid_rows, valid_token_positions]
                        # Make active tokens unmasked
                        final_mask_2d[valid_rows, i, valid_tokens] = 0.0
                        # >>> WE UNMASK THE OBSERVATION WINDOW <<<
                        final_mask_2d[:, i, obs_start : i + 1] = 0.0
                        # >>> WE UNMASK THE OBSERVATION WINDOW <<<
                            
                        final_mask = final_mask_2d.view(bsz, num_heads, q_len, kv_seq_len)
                        final_mask[:, :, :, :min_sparse_index] = 0.0
                        self.final_mask_investigate = final_mask
                        self.snapkv_cache = final_mask[:, :, -1, :].clone().unsqueeze(2)
                        attn_weights = attn_weights + final_mask
                else:
                    # layer_idx == 0 => no pruning
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            elif evalmode == "h2o_true":
                if self.layer_idx > 0:
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    bsz, num_heads, q_len, kv_seq_len = attn_weights.size()
                    final_mask = torch.full_like(attn_weights, float('-inf'))
                    combined_bh = bsz * num_heads
                    final_mask = final_mask.view(combined_bh, q_len, kv_seq_len)
                    max_budget = max(int(key_len * self.sparse_aggression), min_sparse_index)
                
                    final_mask = torch.full_like(attn_weights, float('-inf'))
                    final_mask_2d = final_mask.view(combined_bh, q_len, kv_seq_len)
                    attn_weights_2d = attn_weights.view(combined_bh, q_len, kv_seq_len)

                    active_tokens = torch.full((combined_bh, max_budget), 0, dtype=torch.long, device=attn_weights.device)
                    active_tokens[:, 0] = 0
                    active_counts = torch.ones(combined_bh, dtype=torch.long, device=attn_weights.device)

                    final_mask_2d[:, 0, 0] = 0.0

                    for i in range(1, q_len):
                        kv_cache_budget = torch.full((combined_bh,),
                                                    max(min_sparse_index, int((i + 1 - self.sliding_window - min_sparse_index) * self.sparse_aggression)),
                                                    device=attn_weights.device)
                        row_weights = attn_weights_2d[:, i, :i + 1]
                        can_add = active_counts < kv_cache_budget
                        add_indices = can_add.nonzero(as_tuple=False).squeeze(-1)
                        active_tokens[add_indices, active_counts[add_indices]] = i
                        active_counts[add_indices] += 1

                        cannot_add = ~can_add
                        if cannot_add.any():
                            replace_indices = cannot_add.nonzero(as_tuple=False).squeeze(-1)
                            max_k = kv_cache_budget[replace_indices].max().item()
                            current_active = active_tokens[replace_indices, :max_k]
                            active_importances = row_weights[replace_indices].gather(1, current_active)
                            min_vals, min_idxs = torch.min(active_importances, dim=1, keepdim=True)
                            new_importance = row_weights[replace_indices, i].unsqueeze(1)
                            should_replace = new_importance > min_vals
                            rows_to_replace = replace_indices[should_replace.squeeze(1)]
                            pos_to_replace = min_idxs[should_replace.squeeze(1)].squeeze(1)
                            active_tokens[rows_to_replace, pos_to_replace] = i

                        valid_positions = torch.arange(max_budget, device=attn_weights.device).unsqueeze(0) < active_counts.unsqueeze(1)
                        valid_rows = valid_positions.nonzero(as_tuple=True)[0]
                        valid_token_positions = valid_positions.nonzero(as_tuple=True)[1]
                        valid_tokens = active_tokens[valid_rows, valid_token_positions]
                        final_mask_2d[valid_rows, i, valid_tokens] = 0.0

                    final_mask = final_mask_2d.view(bsz, num_heads, q_len, kv_seq_len)
                    final_mask[:, :, :, :min_sparse_index] = 0.0
                    if self.sliding_window is not None:
                        if not hasattr(self, "window_cache"):
                            self.window_cache = SlidingWindowCache(max_seq_len=1024,
                                                                sliding_window=self.sliding_window,
                                                                device=final_mask.device)
                        window = self.window_cache.get_window(q_len, key_len)
                        final_mask = enforce_sliding_window(final_mask, window)
                    self.final_mask_investigate = final_mask
                    attn_weights = attn_weights + final_mask
                else:
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

            elif evalmode == "quest":
                if self.layer_idx > 0:
                    # Look at https://github.com/mit-han-lab/Quest/blob/main/evaluation/quest_attention.py
                    # Adapted a lot from there.
                    num_tok_per_page = self.num_tok_per_page
                    num_full_pages = q_len // num_tok_per_page
                    if num_full_pages > 0:
                        remaining_tokens = q_len % num_tok_per_page
                        total_pages = num_full_pages + (1 if remaining_tokens > 0 else 0)
                        key_states_full = key_states[:, :, :num_full_pages * num_tok_per_page]
                        key_states_full = key_states_full.transpose(-2, -1).view(
                            bsz, self.num_heads, -1, num_full_pages, num_tok_per_page
                        )
                        key_states_full = key_states_full.amax(dim=-1)  # Take the maximum in each chunk
                        if remaining_tokens > 0:
                            key_states_partial = key_states[:, :, num_full_pages * num_tok_per_page:]
                            pad_size = num_tok_per_page - remaining_tokens
                            key_states_partial = F.pad(key_states_partial, (0, 0, 0, pad_size), value=torch.finfo(key_states.dtype).min)
                            key_states_partial = key_states_partial.transpose(-2, -1).view(
                                bsz, self.num_heads, -1, 1, num_tok_per_page
                            ).amax(dim=-1)  # Take the maximum in the partial page
                            key_states_to_page = torch.cat([key_states_full, key_states_partial], dim=-1)
                            num_pages = num_full_pages +  1
                        else:
                            key_states_to_page = key_states_full  # [B, H, key_len_new, num_full_pages, 2]
                            num_pages = num_full_pages

                        sign = (query_states > 0) + (~(query_states > 0)) * -1
                        key_states_signed = key_states * sign
                        query_states_signed = query_states * sign
                        key_states_reshaped = key_states_to_page.view(bsz, self.num_heads, -1, num_pages)  # Reshape for interaction
                        quest_page_weights = torch.matmul(query_states_signed, key_states_reshaped) / math.sqrt(self.head_dim)
                        quest_page_weights_repeated = quest_page_weights.repeat_interleave(
                            num_tok_per_page, dim=-1
                        )  # [B, H, q_len, key_len_new * num_tok_per_page]
                        quest_page_weights_repeated = quest_page_weights_repeated[..., :key_len]  # Trim excess padding
                        sorted_indices = torch.argsort(
                            quest_page_weights_repeated + attention_mask.view(1, 1, q_len, key_len).float(),
                            dim=-1,
                            descending=True,
                        )  # [B, H, q_len, key_len]
                    else:
                        # initialize random torch tensor [bsz, num_heads, q_len, key_len]
                        importance_mask = torch.softmax(torch.rand(bsz, self.num_heads, q_len, q_len).to(query_states.device) + attention_mask, dim=-1, dtype=torch.float32)
                        # No quest-token mask can exist, so drop tokens randomly.
                        _, sorted_indices = importance_mask.sort(dim=-1, descending=False)

                    sorted_indices = sorted_indices[:, :, -q_len:, :]
                    mask_tensor = sorted_index_to_mask(sorted_indices, attention_mask, min_sparse_index, bsz, q_len, key_len, self.sparse_aggression, self.sliding_window)
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    final_mask = mask_tensor
                    if self.sliding_window is not None:
                        if not hasattr(self, "window_cache"):
                            self.window_cache = SlidingWindowCache(max_seq_len=1024,
                                                                sliding_window=self.sliding_window,
                                                                device=final_mask.device)
                        window = self.window_cache.get_window(q_len, key_len)
                        final_mask = enforce_sliding_window(final_mask, window)

                    self.final_mask_investigate = final_mask
                    attn_weights = attn_weights + mask_tensor + attention_mask
                else:
                    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            else:
                raise ValueError(f"Unknown eval mode {evalmode}")

        if q_len != 1:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if final_mask is not None:
            if self.effective_sparsity is None:
                true_mask = final_mask + attention_mask
                num_deact = true_mask.bool().sum(dim=-1)                   # Number of tokens disabled.
                causally_deact = (attention_mask.bool()).sum(dim=-1).expand_as(num_deact)        # Number of tokens disabled causally anyway
                additional_deact = (num_deact - causally_deact)
                num_active = (~attention_mask.bool()).sum(dim=-1).expand_as(num_deact)    # Number of tokens active at this position if zero-sparsity
                effective_sparsity = 100 * (additional_deact.float() / num_active.float()).mean().item()
                self.effective_sparsity = effective_sparsity
                print("Effective Sparsity:", effective_sparsity, "%\t Sequence Length:", q_len)

        if self.layer_idx == 0:
            if self.effective_sparsity is None:
                self.effective_sparsity = 0.0

        if evalmode == "init_oracle":
            if self.layer_idx == 0:
                self.init_token_importance = torch.softmax(attn_weights.detach().float() + attention_mask, dim=-1, dtype=torch.float32)

        if evalmode == "lookahead_oracle":
            if self.layer_idx == 0:
                self.init_token_importance = torch.softmax(attn_weights.detach().float() + attention_mask, dim=-1, dtype=torch.float32)
            else:
                self.producer.init_token_importance = torch.softmax(attn_weights.detach().float() + attention_mask, dim=-1, dtype=torch.float32)

        if self.inference_mode:
            if "lookahead" in evalmode:
                if self.layer_idx == 0:
                    self.actmagn_masklist[self.layer_idx] = attn_weights.detach().float().sum(dim=2).unsqueeze(dim=2)
                else:
                    self.producer.actmagn_masklist[self.layer_idx] = attn_weights.detach().float().sum(dim=2).unsqueeze(dim=2)
                    
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if use_cache:
            if evalmode == "h2o_true":
                raise ValueError("h2o_true mode is not supported with cache")
        else:
            past_key_value = None

        if not output_attentions:
            attn_weights = None
        # if self.layer_idx == 0:
        # self.tactic_prefill(key_states)
        
        return attn_output, attn_weights

def convert_kvcache_experimental(model, config, producer_frequency):
    producer_layer = None
    producer_layer_device = None
    layer_counter = {'idx': 0}

    def recurse_convert(parent_module):
        nonlocal producer_layer
        nonlocal producer_layer_device
        for name, module in parent_module._modules.items():
            if len(list(module.children())) > 0:
                recurse_convert(module)
            if isinstance(module, LlamaAttention):
                device = next(module.parameters()).device
                dtype = next(module.parameters()).dtype
                if layer_counter['idx'] % producer_frequency == 0:
                    new_module = LlamaAttentionExperimental(config).to(dtype).to(device)
                    producer_layer = new_module
                    producer_layer_device = device
                else:
                    new_module = LlamaAttentionExperimental(
                        config,
                        producer=producer_layer,
                        layer_idx=layer_counter['idx']
                    ).to(dtype).to(device)
                new_module.load_state_dict(module.state_dict(), strict=False)
                is_producer = layer_counter['idx'] % producer_frequency == 0
                if is_producer:
                    print(f"Converted Producer layer '{name}' to LlamaAttentionExperimental at layer index {layer_counter['idx']}")
                else:
                    print(f"Converted layer '{name}' to LlamaAttentionExperimental at layer index {layer_counter['idx']}")
                parent_module._modules[name] = new_module
                layer_counter['idx'] += 1
    recurse_convert(model)
    producer_layer = producer_layer.to(producer_layer_device)
    return model

            # elif evalmode == "tactic_sim":
            #     attn_weights = torch.matmul(
            #         query_states, key_states.transpose(-2, -1)
            #     ) / math.sqrt(self.head_dim)
            #     groups = self.num_key_value_groups
            #     if self.layer_idx == 0:
            #         # keep layer‑0 dense – many sparse papers do that
            #         pass
            #     else:
            #         P = getattr(self, "tactic_threshold", 0.99)

            #         cents = self.producer.cents            # [B, kvH, C, D]
            #         cid   = self.producer.cluster_id       # [B, kvH, T]

            #         kvH      = cents.size(1)               # 24   (key/value heads)
            #         heads_q  = query_states.size(1)        # 24   (query heads)
            #         groups_q = heads_q // kvH              # 1    (must divide exactly)

            #         cid_rep   = cid.repeat_interleave(groups_q, dim=1)  # (B, 72, T)
            #         cluster_sz = getattr(self, "cluster_sz", 32)        # tokens per cluster
            #         C          = cents.size(2)                          # clusters per head

            #         B, _, T, D = key_states.shape
            #         device, dt = key_states.device, key_states.dtype

            #         final_mask  = torch.full((B, heads_q, q_len, T),
            #                                 float('-inf'), device=device, dtype=dt)
            #         attn_weights = final_mask.clone()

            #         # ❷ A/x + b parameters (one per head) initialised from pre‑fill
            #         sample   = torch.randperm(T, device=device)[: max(1, int(0.02 * T))]
            #         k_samp   = key_states[:, :, sample, :]                    # [B, kvH, S, D]

            #         q0 = query_states[:, :kvH, 0:1, :].expand(-1, kvH, len(sample), -1)
            #         logits_sample = (q0 * k_samp).sum(-1).float()             # [B, kvH, S]
            #         logits_sample = logits_sample.sort(-1).values             # ascending

            #         ranks  = torch.arange(1, logits_sample.size(-1) + 1, device=device).float()
            #         prob   = torch.softmax(logits_sample, -1)
            #         inv_r  = 1.0 / ranks
            #         A      = ((prob - prob.mean(-1, keepdim=True)) * inv_r).sum(-1)
            #         B_par  = prob.mean(-1) - A * inv_r.mean()                 # [B, kvH]

            #         # repeat to full num_heads
            #         A     = A.repeat_interleave(groups_q, dim=1)              # (B, 72)
            #         B_par = B_par.repeat_interleave(groups_q, dim=1)          # (B, 72)

            #         H_R = torch.cumsum(
            #             1.0 / torch.arange(1, C + 1, device=device, dtype=dt), dim=0
            #         )                                                         # (C,)

            #         cent_rep = cents.repeat_interleave(groups_q, dim=1)       # (B, 72, C, D)

            #         for t in range(q_len):
            #             q_t = query_states[:, :, t : t + 1]                   # (B, 72, 1, D)

            #             # --- similarity to cluster centroids ----------------------------
            #             c_scores = torch.matmul(q_t, cent_rep.transpose(-2, -1)).squeeze(2)
            #             c_idx    = c_scores.argsort(-1, descending=True)      # (B, 72, C)

            #             rhs   = A.unsqueeze(-1) * H_R + B_par.unsqueeze(-1)   # (B, 72, C)
            #             R_est = (rhs < (1 - P)).sum(-1).add_(1)               # (B, 72)
            #             N_est = (R_est * cluster_sz).clamp_max_(t + 1)        # (B, 72)

            #             # --- build sparse mask ------------------------------------------
            #             row_mask = torch.full((B, heads_q, 1, T),
            #                                 float('-inf'), device=device, dtype=dt)

            #             # *** FIX: restrict token matching to top‑R_est clusters ***
            #             C_range        = torch.arange(C, device=device)                             # (C,)
            #             keep_clusters  = C_range.unsqueeze(0).unsqueeze(0) < R_est.unsqueeze(-1)     # (B, 72, C)
            #             cluster_match  = (cid_rep.unsqueeze(-2) == c_idx.unsqueeze(-1))              # (B, 72, C, T)
            #             gather_mask    = (cluster_match & keep_clusters.unsqueeze(-1)).any(-2)       # (B, 72, T)

            #             rank_in_mask = torch.cumsum(gather_mask.to(torch.int32), dim=-1)             # (B, 72, T)
            #             keep_cond    = gather_mask & (rank_in_mask <= N_est.unsqueeze(-1))           # (B, 72, T)

            #             row_mask.masked_fill_(keep_cond.unsqueeze(2), 0.0)

            #             # --- apply mask --------------------------------------------------
            #             logits = torch.matmul(q_t, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            #             logits += row_mask                                               # simpler, full mask

            #             final_mask[:, :, t : t + 1, :] = row_mask
            #             attn_weights[:, :, t : t + 1, :] = logits

            #         attn_weights = attn_weights + attention_mask + final_mask
            # elif evalmode == "tactic":       # ⬅️ new block
            #     # -- 1. dense raw logits  ----------------------
            #     logits = torch.matmul(query_states,
            #                         key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            #     logits = logits + attention_mask    # causal
                
            #     # -- 2. soft‑max per head ----------------------
            #     probs  = torch.softmax(logits.float(), dim=-1).to(value_states.dtype)
                
            #     # -- 3. build mask -----------------------------
            #     keep_mask = torch.zeros_like(probs, dtype=torch.bool)
                
            #     # vectorised cumulative sum over last dim
            #     cum_probs, indices = torch.sort(probs, dim=-1, descending=True)
            #     cum_probs = torch.cumsum(cum_probs, dim=-1)
            #     keep_flag = cum_probs < tactic_P
            #     # always keep first token where cum≥P
            #     keep_flag[..., 0] = True
            #     # scatter back to original positions
            #     keep_mask.scatter_(-1, indices, keep_flag)
            #     # convert bool→mask (0 or −inf)
            #     mask_tensor = torch.where(keep_mask, 0.,
            #                             torch.tensor(float("-inf"), device=probs.device))
            #     attn_weights = logits + mask_tensor      # reuse scaled logits
            #     final_mask   = mask_tensor
            #     self.final_mask_investigate = final_mask