"""
DSA-style (DeepSeek Sparse Attention) KV Cache for GQA models.

Faithful to the original DeepSeek-V3.2 design:
- Each layer runs its own independent lightning indexer
- Each layer selects its own top-k tokens (no cross-layer sharing)
- Single global ranking per layer (shared across all attention heads)

Buffer layout (same as KeySifterCache):
  [sink_tokens | local_window (circular) | sparse_selected]
"""

import torch
import torch.nn as nn
import math
from contextlib import nullcontext


class LightningIndexer(nn.Module):
    """
    DSA Lightning Indexer with per-layer parameters.

    Each layer has its own q_proj, k_proj, and head_weights.
    Score = sum_h w_h * ReLU(q_h . k_h)
    """

    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        num_indexer_heads: int = 8,
        indexer_head_dim: int = 64,
        producer_frequency: int = 4,  # kept for API compat but not used in per-layer mode
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_indexer_heads = num_indexer_heads
        self.indexer_head_dim = indexer_head_dim
        self.producer_frequency = producer_frequency
        self.indexer_total_dim = num_indexer_heads * indexer_head_dim

        # Per-layer query projection: hidden_state -> indexer queries
        self.q_proj = nn.ModuleList([
            nn.Linear(hidden_size, self.indexer_total_dim, bias=False)
            for _ in range(num_hidden_layers)
        ])

        # Per-layer key projection: hidden_state -> indexer keys
        self.k_proj = nn.ModuleList([
            nn.Linear(hidden_size, self.indexer_total_dim, bias=False)
            for _ in range(num_hidden_layers)
        ])

        # Per-layer head weights
        self.head_weights = nn.Parameter(
            torch.randn(num_hidden_layers, num_indexer_heads, device=device, dtype=dtype) * 0.02
        )

        self.to(device=device, dtype=dtype)

    def project_key(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Project hidden states to indexer key space for a specific layer."""
        k = self.k_proj[layer_idx](hidden_states)
        B, L, _ = k.shape
        return k.view(B, L, self.num_indexer_heads, self.indexer_head_dim)

    def score(self, hidden_states: torch.Tensor, k_index_cache: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Compute importance scores for a specific layer."""
        q = self.q_proj[layer_idx](hidden_states)
        B = q.shape[0]
        q = q.view(B, 1, self.num_indexer_heads, self.indexer_head_dim)

        dots = torch.einsum("bqhd,bthd->bth", q, k_index_cache)
        dots = torch.relu(dots)

        w = self.head_weights[layer_idx]
        scores = torch.einsum("bth,h->bt", dots, w)
        return scores


class DSACache:
    """
    DeepSeek Sparse Attention KV cache — faithful to original design.

    Key differences from KeySifterCache:
    - Per-layer indexer (every layer runs its own scoring independently)
    - No producer/consumer grouping for selection
    - Each layer has its own indexer key cache and selects its own top-k
    - Single global ranking per layer (shared across attention heads)
    """

    def __init__(
        self,
        config: object,
        indexer: LightningIndexer,
        batch_size: int = 1,
        max_length: int = 32 * 1024,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size: int = 8,
        producer_frequency: int = 4,  # kept for API compat
        local_window: int = 512,
        min_sparse_index: int = 128,
        predict_interval: int = 1,
        enable_neighbor_fetch: bool = False,
    ) -> None:
        self.config = config
        self.indexer = indexer
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype

        self.num_hidden_layers = config.num_hidden_layers
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_attention_heads

        self.sparse_budget = sparse_budget
        self.chunk_size = chunk_size
        self.producer_frequency = producer_frequency
        self.predict_interval = predict_interval
        self.enable_neighbor_fetch = enable_neighbor_fetch
        self.effective_sparse_capacity = sparse_budget * 2 if enable_neighbor_fetch else sparse_budget

        self.local_window = local_window
        self.min_sparse_index = min_sparse_index

        # Main KV cache: [layers, B, H_kv, max_length, head_dim]
        self.v_cache = torch.zeros(
            self.num_hidden_layers, batch_size, self.num_key_value_heads,
            max_length, self.head_dim, dtype=dtype, device=device,
        )
        self.k_cache = torch.zeros(
            self.num_hidden_layers, batch_size, self.num_key_value_heads,
            max_length, self.head_dim, dtype=dtype, device=device,
        )

        # Per-layer indexer key cache: [layers, B, max_length, H_i, d_i]
        self.k_index_cache = torch.zeros(
            self.num_hidden_layers, batch_size, max_length,
            indexer.num_indexer_heads, indexer.indexer_head_dim,
            dtype=dtype, device=device,
        )

        # Per-layer buffer: [layers, B, H_kv, buffer_size, head_dim]
        buffer_size = min_sparse_index + local_window + self.effective_sparse_capacity
        self.k_cache_buffer = torch.zeros(
            self.num_hidden_layers, batch_size, self.num_key_value_heads,
            buffer_size, self.head_dim, dtype=dtype, device=device,
        )
        self.v_cache_buffer = torch.zeros(
            self.num_hidden_layers, batch_size, self.num_key_value_heads,
            buffer_size, self.head_dim, dtype=dtype, device=device,
        )

        # Per-layer tracking of how many sparse tokens are in the buffer
        self._num_sparse_selected = [0] * self.num_hidden_layers

        # State tracking
        self.kv_offset = 0
        self.prefill_len = 0
        self.gen_offset = 0
        self.local_window_head = 0
        self._dense_decode_cutoff = 0
        self._last_num_sparse_selected = 0  # for layer 0 compat
        self._force_next_prediction = False
        self.prefill_cont_dense = True

        self._uncommitted_decode = False
        self._uncommitted_incoming = 0

        self.profiler = None
        self.copy_stream = torch.cuda.Stream(device=device)

    def _kv_len_eff(self) -> int:
        if self._uncommitted_decode and self._uncommitted_incoming > 0:
            return self.kv_offset + self._uncommitted_incoming
        return self.kv_offset

    def _record(self, name):
        if self.profiler is not None:
            return self.profiler.record(name)
        return nullcontext()

    def get_kv_len(self):
        return self.kv_offset

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.k_index_cache.zero_()
        self.k_cache_buffer.zero_()
        self.v_cache_buffer.zero_()
        self._num_sparse_selected = [0] * self.num_hidden_layers
        self.kv_offset = 0
        self.prefill_len = 0
        self.gen_offset = 0
        self.local_window_head = 0
        self._dense_decode_cutoff = 0
        self._last_num_sparse_selected = 0
        self._force_next_prediction = False
        self._uncommitted_decode = False
        self._uncommitted_incoming = 0

    def print_stats(self):
        print(f"DSACache | sparse_budget {self.sparse_budget} | per_layer_indexer | "
              f"indexer_heads {self.indexer.num_indexer_heads} | indexer_dim {self.indexer.indexer_head_dim} | "
              f"cached {self.kv_offset}")

    def H2D(self):
        pass

    def get_svd(self, key_states: torch.Tensor, layer_idx: int, fake_svd: bool = False):
        pass

    def _hidden_from_keys(self, key_states: torch.Tensor) -> torch.Tensor:
        """Approximate hidden states from key states for indexer projection.
        key_states: [B, H_kv, L, head_dim] -> [B, L, hidden_size]
        """
        B, H, L, D = key_states.shape
        k_flat = key_states.permute(0, 2, 1, 3).reshape(B, L, H * D)
        hidden_size = self.indexer.hidden_size
        if k_flat.shape[-1] < hidden_size:
            repeats = (hidden_size // k_flat.shape[-1]) + 1
            k_flat = k_flat.repeat(1, 1, repeats)[:, :, :hidden_size]
        elif k_flat.shape[-1] > hidden_size:
            k_flat = k_flat[:, :, :hidden_size]
        return k_flat

    def prefill_kv_cache(self, new_v_cache, layer_idx, key_states_roped, query=None):
        current_len = self.prefill_len
        incoming = key_states_roped.shape[2]
        new_total_len = current_len + incoming

        self.k_cache[layer_idx, :, :, current_len:new_total_len] = key_states_roped
        self.v_cache[layer_idx, :, :, current_len:new_total_len] = new_v_cache

        # Build per-layer indexer key cache
        h_approx = self._hidden_from_keys(key_states_roped)
        k_index = self.indexer.project_key(h_approx, layer_idx)
        self.k_index_cache[layer_idx, :, current_len:new_total_len] = k_index

        # Sink tokens
        if current_len == 0:
            sink_len = min(self.min_sparse_index, incoming)
            if sink_len > 0:
                self.k_cache_buffer[layer_idx, :, :, :sink_len] = key_states_roped[:, :, :sink_len]
                self.v_cache_buffer[layer_idx, :, :, :sink_len] = new_v_cache[:, :, :sink_len]

        # Local window
        lw = self.local_window
        buf_start = self.min_sparse_index
        if incoming >= lw:
            self.k_cache_buffer[layer_idx, :, :, buf_start:buf_start+lw] = key_states_roped[:, :, -lw:]
            self.v_cache_buffer[layer_idx, :, :, buf_start:buf_start+lw] = new_v_cache[:, :, -lw:]
        else:
            self.k_cache_buffer[layer_idx, :, :, buf_start:buf_start+incoming] = key_states_roped
            self.v_cache_buffer[layer_idx, :, :, buf_start:buf_start+incoming] = new_v_cache

        if layer_idx == self.num_hidden_layers - 1:
            self.prefill_len = new_total_len
            self.kv_offset = new_total_len
            self.gen_offset = 0
            self._dense_decode_cutoff = new_total_len + 1
            self._uncommitted_decode = False
            self._uncommitted_incoming = 0

    def update_kv_cache(self, new_k_cache, new_v_cache, layer_idx):
        kv_offset = self.kv_offset
        incoming = 1

        if layer_idx == 0:
            self._uncommitted_decode = True
            self._uncommitted_incoming = incoming

        self.k_cache[layer_idx, :, :, kv_offset:kv_offset+incoming] = new_k_cache
        self.v_cache[layer_idx, :, :, kv_offset:kv_offset+incoming] = new_v_cache

        buf_pos = self.min_sparse_index + self.local_window_head
        self.k_cache_buffer[layer_idx, :, :, buf_pos:buf_pos+incoming] = new_k_cache
        self.v_cache_buffer[layer_idx, :, :, buf_pos:buf_pos+incoming] = new_v_cache

        # Build indexer key cache for this layer
        h_approx = self._hidden_from_keys(new_k_cache)
        k_index = self.indexer.project_key(h_approx, layer_idx)
        self.k_index_cache[layer_idx, :, kv_offset:kv_offset+incoming] = k_index

        if layer_idx == self.num_hidden_layers - 1:
            self.local_window_head = (self.local_window_head + incoming) % self.local_window
            self.kv_offset += incoming
            self.gen_offset += incoming
            self._uncommitted_decode = False
            self._uncommitted_incoming = 0
            self._force_next_prediction = False

    def prefetch_single_layer(self, hidden_states, layer_idx):
        """Score and gather for a single layer. Called at every layer (faithful DSA)."""
        kv_len = self._kv_len_eff()
        if kv_len <= self._dense_decode_cutoff:
            return
        # Layer 0 is always dense
        if layer_idx == 0:
            return

        # Prediction stride
        if self.predict_interval > 1 and not self._force_next_prediction:
            non_dense_offset = self.gen_offset - 1
            if non_dense_offset > 0 and non_dense_offset % self.predict_interval != 0:
                return

        if hidden_states.shape[1] > 1:
            hidden_states = hidden_states[:, -1:, :]

        limit = self.kv_offset

        # Score with this layer's indexer
        k_index = self.k_index_cache[layer_idx, :, :limit]
        scores = self.indexer.score(hidden_states, k_index, layer_idx)

        # Mask sink and local window
        if self.min_sparse_index > 0:
            sink = min(self.min_sparse_index, limit)
            scores[:, :sink] = float("-inf")

        local_start = max(0, kv_len - self.local_window)
        if local_start < limit:
            scores[:, local_start:] = float("-inf")

        selection_end = min(local_start, limit)
        selection_start = self.min_sparse_index
        num_available = max(0, selection_end - selection_start)
        num_to_select = min(self.sparse_budget, num_available)

        if num_to_select > 0:
            _, topk_indices = torch.topk(scores, k=num_to_select, dim=-1)
            topk_indices, _ = topk_indices.sort(dim=-1)

            sparse_start = self.min_sparse_index + self.local_window

            # Gather for this single layer
            idx = topk_indices.unsqueeze(1).unsqueeze(-1)
            idx = idx.expand(-1, self.num_key_value_heads, -1, self.head_dim)

            k_source = self.k_cache[layer_idx]
            v_source = self.v_cache[layer_idx]

            out_k = self.k_cache_buffer[layer_idx, :, :, sparse_start:sparse_start+num_to_select]
            torch.gather(k_source, dim=2, index=idx, out=out_k)

            out_v = self.v_cache_buffer[layer_idx, :, :, sparse_start:sparse_start+num_to_select]
            torch.gather(v_source, dim=2, index=idx, out=out_v)

            self._num_sparse_selected[layer_idx] = num_to_select
        else:
            self._num_sparse_selected[layer_idx] = 0

    def prefetch_layer_group(self, hidden_states, start_layer_idx):
        """Kept for API compatibility but DSA uses prefetch_single_layer."""
        pass

    def get_retrieval_position_ids(self, layer_idx=None, query_states=None):
        return None

    def get_key_cache(self, layer_idx, position_ids=None, rope_func=None, cos_sin_cache=None):
        kv_len = self._kv_len_eff()
        if layer_idx == 0 or kv_len <= self._dense_decode_cutoff:
            return self.k_cache[layer_idx, :, :, :kv_len].clone()

        num_selected = self._num_sparse_selected[layer_idx]
        total_len = self.min_sparse_index + self.local_window + num_selected
        return self.k_cache_buffer[layer_idx, :, :, :total_len]

    def get_value_cache(self, layer_idx, position_ids=None):
        kv_len = self._kv_len_eff()
        if layer_idx == 0 or kv_len <= self._dense_decode_cutoff:
            return self.v_cache[layer_idx, :, :, :kv_len].clone()

        num_selected = self._num_sparse_selected[layer_idx]
        total_len = self.min_sparse_index + self.local_window + num_selected
        return self.v_cache_buffer[layer_idx, :, :, :total_len]
