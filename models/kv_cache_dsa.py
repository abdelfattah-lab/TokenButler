"""
DSA-style (DeepSeek Sparse Attention) KV Cache for GQA models.

Implements a lightning-indexer-based sparse attention cache compatible with
Llama-3.1-8B's GQA attention. The indexer is a lightweight set of projection
heads that score every KV entry, selects top-k, and gathers them into a
pre-assembled buffer for flash attention.

This is designed for efficiency benchmarking -- weights can be randomly
initialized since only the compute graph and memory access patterns matter.

Buffer layout (identical to KeySifterCache):
  [sink_tokens | local_window (circular) | sparse_selected]
"""

import torch
import torch.nn as nn
import math
from contextlib import nullcontext


class LightningIndexer(nn.Module):
    """
    DSA Lightning Indexer: lightweight scoring heads for KV importance.

    Produces a scalar relevance score for each (query, key) pair by:
      1. Project query hidden state -> indexer query vectors
      2. Project key hidden state -> indexer key vectors (done during prefill/update)
      3. Score = sum_h w_h * ReLU(q_h . k_h)

    Architecture choices following DSA:
      - num_indexer_heads: 64 small heads for scoring (independent of attention heads)
      - indexer_head_dim: 128 per head
      - Shared across all attention heads (single global ranking)
      - Per-layer indexer (each layer scores independently)
    """

    def __init__(
        self,
        hidden_size: int,           # Model hidden size (4096 for Llama-8B)
        num_hidden_layers: int,     # Total model layers
        num_indexer_heads: int = 8,
        indexer_head_dim: int = 64,
        producer_frequency: int = 4,
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

        num_producers = math.ceil(num_hidden_layers / producer_frequency)

        # Per-producer query projection: hidden_state -> indexer queries
        # Each producer serves producer_frequency consumer layers, but the indexer
        # produces a single global ranking shared across all layers in the group.
        self.q_proj = nn.ModuleList([
            nn.Linear(hidden_size, self.indexer_total_dim, bias=False)
            for _ in range(num_producers)
        ])

        # Per-producer key projection: hidden_state -> indexer keys
        # Applied during prefill/update to build the indexer KV cache
        self.k_proj = nn.ModuleList([
            nn.Linear(hidden_size, self.indexer_total_dim, bias=False)
            for _ in range(num_producers)
        ])

        # Per-head learned scalar weights for combining head scores
        # Shape: [num_producers, num_indexer_heads]
        self.head_weights = nn.Parameter(
            torch.randn(num_producers, num_indexer_heads, device=device, dtype=dtype) * 0.02
        )

        self.to(device=device, dtype=dtype)

    def project_key(self, hidden_states: torch.Tensor, producer_idx: int) -> torch.Tensor:
        """Project hidden states to indexer key space.

        Args:
            hidden_states: [B, L, hidden_size]
            producer_idx: which producer (0-indexed)

        Returns:
            [B, L, num_indexer_heads, indexer_head_dim]
        """
        # [B, L, indexer_total_dim]
        k = self.k_proj[producer_idx](hidden_states)
        B, L, _ = k.shape
        return k.view(B, L, self.num_indexer_heads, self.indexer_head_dim)

    def score(self, hidden_states: torch.Tensor, k_index_cache: torch.Tensor, producer_idx: int) -> torch.Tensor:
        """Compute importance scores for all cached keys.

        Args:
            hidden_states: [B, 1, hidden_size] - current decode token
            k_index_cache: [B, T, num_indexer_heads, indexer_head_dim] - cached indexer keys
            producer_idx: which producer

        Returns:
            [B, T] - scalar score per cached position
        """
        # Project query: [B, 1, indexer_total_dim] -> [B, 1, H_i, d_i]
        q = self.q_proj[producer_idx](hidden_states)
        B = q.shape[0]
        q = q.view(B, 1, self.num_indexer_heads, self.indexer_head_dim)

        # Dot product per head: [B, 1, H_i, d_i] x [B, T, H_i, d_i] -> [B, T, H_i]
        # einsum: b1hd, bthd -> bth
        dots = torch.einsum("bqhd,bthd->bth", q, k_index_cache)

        # ReLU (DSA uses ReLU to ensure non-negative scores)
        dots = torch.relu(dots)

        # Weighted sum across heads: [B, T, H_i] x [H_i] -> [B, T]
        w = self.head_weights[producer_idx]  # [H_i]
        scores = torch.einsum("bth,h->bt", dots, w)

        return scores


class DSACache:
    """
    DeepSeek Sparse Attention style KV cache for GQA models.

    Key differences from KeySifterCache:
    - Uses a LightningIndexer (small projection heads) instead of an MLP predictor
    - Indexer scores every position directly (no importance queries via projected keys)
    - Maintains a separate indexer key cache alongside the main KV cache
    - Single global token ranking shared across all attention heads

    Similarities with KeySifterCache:
    - Same buffer structure: [sink | local_window | sparse_selected]
    - Same producer/consumer layer grouping
    - Same top-k selection + gather pattern
    - Compatible API for drop-in replacement
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
        producer_frequency: int = 4,
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

        # Indexer key cache: [B, max_length, num_indexer_heads, indexer_head_dim]
        # Shared across layers within a producer group (single global ranking)
        num_producers = math.ceil(self.num_hidden_layers / producer_frequency)
        self.k_index_cache = torch.zeros(
            num_producers, batch_size, max_length,
            indexer.num_indexer_heads, indexer.indexer_head_dim,
            dtype=dtype, device=device,
        )

        # Buffer: [layers, B, H_kv, buffer_size, head_dim]
        buffer_size = min_sparse_index + local_window + self.effective_sparse_capacity
        self.k_cache_buffer = torch.zeros(
            self.num_hidden_layers, batch_size, self.num_key_value_heads,
            buffer_size, self.head_dim, dtype=dtype, device=device,
        )
        self.v_cache_buffer = torch.zeros(
            self.num_hidden_layers, batch_size, self.num_key_value_heads,
            buffer_size, self.head_dim, dtype=dtype, device=device,
        )

        # State tracking
        self.kv_offset = 0
        self.prefill_len = 0
        self.gen_offset = 0
        self.local_window_head = 0
        self._dense_decode_cutoff = 0
        self._last_num_sparse_selected = 0
        self._force_next_prediction = False
        self.prefill_cont_dense = True

        self._uncommitted_decode = False
        self._uncommitted_incoming = 0

        # Profiler support
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
        print(f"DSACache | sparse_budget {self.sparse_budget} | producer_freq {self.producer_frequency} | "
              f"indexer_heads {self.indexer.num_indexer_heads} | indexer_dim {self.indexer.indexer_head_dim} | "
              f"cached {self.kv_offset}")

    def H2D(self):
        pass  # GPU-only, no transfer needed

    def get_svd(self, key_states: torch.Tensor, layer_idx: int, fake_svd: bool = False):
        pass  # No-op for API compatibility

    def prefill_kv_cache(
        self,
        new_v_cache: torch.Tensor,
        layer_idx: int,
        key_states_roped: torch.Tensor,
        query=None,
    ):
        """Store prefill KV and build indexer key cache."""
        current_len = self.prefill_len
        incoming = key_states_roped.shape[2]
        new_total_len = current_len + incoming

        # Store RoPEd keys and values
        self.k_cache[layer_idx, :, :, current_len:new_total_len] = key_states_roped
        self.v_cache[layer_idx, :, :, current_len:new_total_len] = new_v_cache

        # Build indexer key cache at producer layers
        # The indexer needs hidden states, but during prefill we don't have them
        # readily accessible per-layer. Instead, we project the key states through
        # the indexer's key projection. This is an approximation -- in real DSA,
        # the indexer uses the hidden states before attention, but for efficiency
        # benchmarking the compute pattern is the same.
        if layer_idx % self.producer_frequency == 0:
            prod_idx = layer_idx // self.producer_frequency
            # key_states_roped: [B, H_kv, L, head_dim] -> we need [B, L, hidden_dim]
            # Approximate: concatenate across KV heads and project
            B, H, L, D = key_states_roped.shape
            # Reshape to [B, L, H*D] and pad/truncate to hidden_size
            k_flat = key_states_roped.permute(0, 2, 1, 3).reshape(B, L, H * D)
            hidden_size = self.indexer.hidden_size
            if k_flat.shape[-1] < hidden_size:
                # Pad by repeating
                repeats = (hidden_size // k_flat.shape[-1]) + 1
                k_flat = k_flat.repeat(1, 1, repeats)[:, :, :hidden_size]
            elif k_flat.shape[-1] > hidden_size:
                k_flat = k_flat[:, :, :hidden_size]

            # Project to indexer key space: [B, L, H_i, d_i]
            k_index = self.indexer.project_key(k_flat, prod_idx)
            self.k_index_cache[prod_idx, :, current_len:new_total_len] = k_index

        # Copy sink tokens to buffer (first chunk only)
        if current_len == 0:
            sink_len = min(self.min_sparse_index, incoming)
            if sink_len > 0:
                self.k_cache_buffer[layer_idx, :, :, :sink_len] = key_states_roped[:, :, :sink_len]
                self.v_cache_buffer[layer_idx, :, :, :sink_len] = new_v_cache[:, :, :sink_len]

        # Initialize local window region with last local_window tokens
        lw = self.local_window
        buf_start = self.min_sparse_index
        if incoming >= lw:
            self.k_cache_buffer[layer_idx, :, :, buf_start:buf_start+lw] = key_states_roped[:, :, -lw:]
            self.v_cache_buffer[layer_idx, :, :, buf_start:buf_start+lw] = new_v_cache[:, :, -lw:]
        else:
            self.k_cache_buffer[layer_idx, :, :, buf_start:buf_start+incoming] = key_states_roped
            self.v_cache_buffer[layer_idx, :, :, buf_start:buf_start+incoming] = new_v_cache

        # Update state on last layer
        if layer_idx == self.num_hidden_layers - 1:
            self.prefill_len = new_total_len
            self.kv_offset = new_total_len
            self.gen_offset = 0
            self._dense_decode_cutoff = new_total_len + 1
            self._uncommitted_decode = False
            self._uncommitted_incoming = 0

    def update_kv_cache(
        self,
        new_k_cache: torch.Tensor,
        new_v_cache: torch.Tensor,
        layer_idx: int,
    ):
        """Store a new decode token."""
        kv_offset = self.kv_offset
        incoming = 1

        if layer_idx == 0:
            self._uncommitted_decode = True
            self._uncommitted_incoming = incoming

        # Store in main cache
        self.k_cache[layer_idx, :, :, kv_offset:kv_offset+incoming] = new_k_cache
        self.v_cache[layer_idx, :, :, kv_offset:kv_offset+incoming] = new_v_cache

        # Store in circular buffer (local window region)
        buf_pos = self.min_sparse_index + self.local_window_head
        self.k_cache_buffer[layer_idx, :, :, buf_pos:buf_pos+incoming] = new_k_cache
        self.v_cache_buffer[layer_idx, :, :, buf_pos:buf_pos+incoming] = new_v_cache

        # Update indexer key cache at producer layers
        if layer_idx % self.producer_frequency == 0:
            prod_idx = layer_idx // self.producer_frequency
            B, H, L, D = new_k_cache.shape
            k_flat = new_k_cache.permute(0, 2, 1, 3).reshape(B, L, H * D)
            hidden_size = self.indexer.hidden_size
            if k_flat.shape[-1] < hidden_size:
                repeats = (hidden_size // k_flat.shape[-1]) + 1
                k_flat = k_flat.repeat(1, 1, repeats)[:, :, :hidden_size]
            elif k_flat.shape[-1] > hidden_size:
                k_flat = k_flat[:, :, :hidden_size]
            k_index = self.indexer.project_key(k_flat, prod_idx)
            self.k_index_cache[prod_idx, :, kv_offset:kv_offset+incoming] = k_index

        # Last layer: commit
        if layer_idx == self.num_hidden_layers - 1:
            old_head = self.local_window_head
            self.local_window_head = (self.local_window_head + incoming) % self.local_window
            self.kv_offset += incoming
            self.gen_offset += incoming
            self._uncommitted_decode = False
            self._uncommitted_incoming = 0
            self._force_next_prediction = False

    def prefetch_layer_group(
        self,
        hidden_states: torch.Tensor,
        start_layer_idx: int,
    ):
        """Score all cached tokens with the lightning indexer and gather top-k."""
        kv_len = self._kv_len_eff()
        if kv_len <= self._dense_decode_cutoff:
            return

        # Prediction stride
        if self.predict_interval > 1 and not self._force_next_prediction:
            non_dense_offset = self.gen_offset - 1
            if non_dense_offset > 0 and non_dense_offset % self.predict_interval != 0:
                return

        if hidden_states.shape[1] > 1:
            hidden_states = hidden_states[:, -1:, :]

        bsz = hidden_states.shape[0]
        prod_idx = start_layer_idx // self.producer_frequency

        # Determine consumer layers
        consumer_start = start_layer_idx + 1
        if consumer_start >= self.num_hidden_layers:
            return
        consumer_end = min(consumer_start + self.producer_frequency, self.num_hidden_layers)
        num_layers_in_group = consumer_end - consumer_start
        if num_layers_in_group <= 0:
            return

        # 1. Score all cached positions with the indexer
        with self._record('indexer_score'):
            # Get indexer keys up to committed length
            limit = self.kv_offset  # Only score committed positions
            k_index = self.k_index_cache[prod_idx, :, :limit]  # [B, limit, H_i, d_i]

            # Score: [B, limit]
            scores = self.indexer.score(hidden_states, k_index, prod_idx)

        # 2. Mask sink and local window positions
        with self._record('mask_and_select'):
            if self.min_sparse_index > 0:
                sink = min(self.min_sparse_index, limit)
                scores[:, :sink] = float("-inf")

            local_start = max(0, kv_len - self.local_window)
            if local_start < limit:
                scores[:, local_start:] = float("-inf")

            # 3. Top-k selection
            selection_end = min(local_start, limit)
            selection_start = self.min_sparse_index
            num_available = max(0, selection_end - selection_start)
            num_to_select = min(self.sparse_budget, num_available)

        if num_to_select > 0:
            with self._record('topk_selection'):
                _, topk_indices = torch.topk(scores, k=num_to_select, dim=-1)  # [B, budget]
                topk_indices, _ = topk_indices.sort(dim=-1)

            num_in_buffer = num_to_select
            self._last_num_sparse_selected = num_in_buffer

            # 4. Gather into buffer for all consumer layers
            # topk_indices: [B, num_to_select] -> expand for all layers and heads
            with self._record('gather_kv'):
                sparse_start = self.min_sparse_index + self.local_window

                # Expand indices: [B, num_to_select] -> [num_layers, B, H_kv, num_to_select, head_dim]
                idx = topk_indices.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
                idx = idx.expand(num_layers_in_group, -1, self.num_key_value_heads, -1, self.head_dim)

                k_source = self.k_cache[consumer_start:consumer_end]
                v_source = self.v_cache[consumer_start:consumer_end]

                out_k = self.k_cache_buffer[consumer_start:consumer_end, :, :, sparse_start:sparse_start+num_in_buffer]
                torch.gather(k_source, dim=3, index=idx, out=out_k)

                out_v = self.v_cache_buffer[consumer_start:consumer_end, :, :, sparse_start:sparse_start+num_in_buffer]
                torch.gather(v_source, dim=3, index=idx, out=out_v)
        else:
            self._last_num_sparse_selected = 0

    def get_retrieval_position_ids(self, layer_idx=None, query_states=None):
        return None

    def get_key_cache(self, layer_idx, position_ids=None, rope_func=None, cos_sin_cache=None):
        kv_len = self._kv_len_eff()
        if layer_idx == 0 or kv_len <= self._dense_decode_cutoff:
            return self.k_cache[layer_idx, :, :, :kv_len].clone()

        num_selected = self._last_num_sparse_selected
        total_len = self.min_sparse_index + self.local_window + num_selected
        return self.k_cache_buffer[layer_idx, :, :, :total_len]

    def get_value_cache(self, layer_idx, position_ids=None):
        kv_len = self._kv_len_eff()
        if layer_idx == 0 or kv_len <= self._dense_decode_cutoff:
            return self.v_cache[layer_idx, :, :, :kv_len].clone()

        num_selected = self._last_num_sparse_selected
        total_len = self.min_sparse_index + self.local_window + num_selected
        return self.v_cache_buffer[layer_idx, :, :, :total_len]
