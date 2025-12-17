################################################################################
#
# KeySifter KV Cache for xKV
# 
# This module implements the KeySifterCache class that uses a learned predictor
# for importance-based sparse attention during decoding.
#
################################################################################

import torch
import torch.nn.functional as F
import math
from typing import Optional, Callable

from .keysifter_predictor import KeySifterPredictor


class KeySifterCache:
    """
    KV Cache with KeySifter-based sparse attention for decode.
    
    During prefill:
        - Store full RoPEd K/V cache for retrieval during decode
        - Project RoPEd keys to reduced dimension (k_proj_cache) for efficient importance scoring
    
    During decode:
        - Use predictor to compute importance queries
        - Score against projected key cache (avoids recomputation)
        - Select top-k positions based on importance scores
        - Retrieve RoPEd keys directly (no RoPE computation needed)
    
    This approach trades memory for compute efficiency during long decode sequences.
    """
    
    def __init__(
        self,
        config: object,
        predictor: KeySifterPredictor,
        batch_size: int = 1,
        max_length: int = 32 * 1024,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size: int = 8,
        producer_frequency: int = 4,
    ) -> None:
        """
        Args:
            config: Model configuration
            predictor: Trained KeySifterPredictor instance
            batch_size: Batch size
            max_length: Maximum sequence length
            device: Device to use
            dtype: Data type
            sparse_budget: Number of tokens to select during sparse attention
            chunk_size: Chunk size for selection (for compatibility with ShadowKV interface)
            producer_frequency: Number of layers served by one predictor
        """
        self.config = config
        self.predictor = predictor
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
        self.dDash = predictor.dDash
        
        # Local window to always include (similar to ShadowKV)
        self.local_window = 32  # Always include last N tokens
        
        # Full value cache (CPU for large contexts, GPU for small)
        self.v_cache = torch.zeros(
            self.num_hidden_layers,
            batch_size,
            self.num_key_value_heads,
            max_length,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        
        # Full key cache (RoPEd - stored during prefill for efficient decode retrieval)
        self.k_cache = torch.zeros(
            self.num_hidden_layers,
            batch_size,
            self.num_key_value_heads,
            max_length,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        
        # Projected key cache for importance scoring
        # Shape: [num_layers, batch, num_key_value_heads, max_length, dDash]
        # Store projections per KV head (8) - will broadcast to attention heads (32) during scoring
        # This saves 4x memory compared to storing per attention head
        self.k_proj_cache = torch.zeros(
            self.num_hidden_layers,
            batch_size,
            self.num_key_value_heads,  # Use KV heads (8), not attention heads (32)
            max_length,
            self.dDash,
            device=device,
            dtype=dtype,
        )
        
        # Buffer for assembling KV during decode: [local_window (circular) | sparse_selected]
        # Only stores recent local_window tokens + space for gathering sparse tokens
        buffer_size = self.local_window + sparse_budget
        self.k_cache_buffer = torch.zeros(
            self.num_hidden_layers,
            batch_size,
            self.num_key_value_heads,
            buffer_size,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        self.v_cache_buffer = torch.zeros(
            self.num_hidden_layers,
            batch_size,
            self.num_key_value_heads,
            buffer_size,
            self.head_dim,
            device=device,
            dtype=dtype,
        )

        # State tracking
        self.kv_offset = 0
        self.prefill_len = 0
        self.gen_offset = 0
        self.last_projected_pos = 0  # Track up to which position we've projected decode tokens
        self.local_window_head = 0  # Circular queue head pointer for local window
        
        # Store importance queries from producer layers
        # Shape: [batch*heads, N_slots, 1, dDash] (for decode, Lq=1)
        self.q_importance_cache = None
        
        # Optional profiler for timing instrumentation (set externally)
        self.profiler = None
        
        self.copy_stream = torch.cuda.Stream()
    
    def _record(self, name):
        """Context manager for profiling. No-op if profiler is None."""
        if self.profiler is not None:
            return self.profiler.record(name)
        else:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()
    
    def print_stats(self):
        print(f"KeySifterCache | sparse_budget {self.sparse_budget} | producer_freq {self.producer_frequency} | dDash {self.dDash} | cached {self.kv_offset}")
    
    def get_kv_len(self):
        return self.kv_offset
    
    def clear(self):
        """Reset cache state for new sequence."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.k_proj_cache.zero_()
        self.k_cache_buffer.zero_()
        self.v_cache_buffer.zero_()
        self.kv_offset = 0
        self.prefill_len = 0
        self.gen_offset = 0
        self.last_projected_pos = 0
        self.local_window_head = 0
        self.q_importance_cache = None

    def H2D(self):
        """Host to device transfer (no-op if already on GPU)."""
        pass
    
    def get_svd(self, key_states: torch.Tensor, layer_idx: int, fake_svd: bool = False):
        """
        No-op for KeySifterCache. Kept for API compatibility with other cache implementations.
        
        Key projection for importance scoring is performed in prefill_kv_cache() on RoPEd keys,
        not here on un-RoPEd keys.
        
        Args:
            key_states: Unused
            layer_idx: Unused
            fake_svd: Unused
        """
        pass
    
    def prefill_kv_cache(
        self,
        new_v_cache: torch.Tensor,
        layer_idx: int,
        key_states_roped: torch.Tensor,
        query: torch.Tensor = None,
    ):
        """
        Store prefill K/V cache and prepare for sparse decode.
        
        Args:
            new_v_cache: [bsz, num_kv_heads, seq_len, head_dim] - value states
            layer_idx: Layer index
            key_states_roped: [bsz, num_kv_heads, seq_len, head_dim] - RoPEd key states
            query: Optional query states (unused)
        """
        seq_len = new_v_cache.shape[2]
        
        # Store full RoPEd key cache (for sparse retrieval during decode)
        self.k_cache[layer_idx, :, :, :seq_len].copy_(key_states_roped)
        
        # Store value cache
        self.v_cache[layer_idx, :, :, :seq_len].copy_(new_v_cache)
        
        # Project RoPEd keys for importance scoring (per KV head)
        # key_states_roped: [B, num_kv_heads, L, head_dim]
        # proj_weight: [num_kv_heads, head_dim, dDash] (aggregated during predictor loading)
        proj_weight = self.predictor.key_cache_proj[layer_idx]  # [num_kv_heads, head_dim, dDash]
        k_proj = torch.einsum("bhlk,hkd->bhld", key_states_roped, proj_weight)
        self.k_proj_cache[layer_idx, :, :, :seq_len].copy_(k_proj)
        
        # Initialize circular buffer with last local_window tokens from prefill
        # These will be the most recent tokens at the start of decode
        local_start = max(0, seq_len - self.local_window)
        local_len = seq_len - local_start
        self.k_cache_buffer[layer_idx, :, :, :local_len].copy_(
            key_states_roped[:, :, local_start:]
        )
        self.v_cache_buffer[layer_idx, :, :, :local_len].copy_(
            new_v_cache[:, :, local_start:]
        )
        # Circular buffer starts with head at 0, and local_len items filled
        # If seq_len < local_window, only first local_len positions are valid
        
        # Only update prefill_len after the last layer to ensure all layers see prefill_len=0
        # during the prefill pass (otherwise subsequent layers think prefill is complete)
        if layer_idx == self.num_hidden_layers - 1:
            self.prefill_len = seq_len
            self.kv_offset = seq_len
            self.gen_offset = 0
            self.last_projected_pos = seq_len  # Prefill tokens are already projected
    
    def update_kv_cache(
        self,
        new_k_cache: torch.Tensor,
        new_v_cache: torch.Tensor,
        layer_idx: int,
    ):
        """
        Update cache with new decode tokens.

        Args:
            new_k_cache: [bsz, num_kv_heads, 1, head_dim] - new RoPEd key
            new_v_cache: [bsz, num_kv_heads, 1, head_dim] - new value
            layer_idx: Layer index
        """
        incoming = new_k_cache.shape[2]

        # Validate cache bounds
        if self.kv_offset + incoming > self.max_length:
            raise RuntimeError(
                f"Sequence length {self.kv_offset + incoming} exceeds max_length {self.max_length}"
            )

        # Add new token to circular buffer (buffer[0:local_window] is the circular queue)
        buffer_pos = self.local_window_head
        self.k_cache_buffer[layer_idx, :, :, buffer_pos:buffer_pos + incoming].copy_(new_k_cache)
        self.v_cache_buffer[layer_idx, :, :, buffer_pos:buffer_pos + incoming].copy_(new_v_cache)

        if layer_idx == self.num_hidden_layers - 1:
            # Update circular buffer head (wraps around within local_window)
            old_head = self.local_window_head
            self.local_window_head = (self.local_window_head + incoming) % self.local_window

            # Update position tracking
            self.kv_offset += incoming
            self.gen_offset += incoming

            # Archive when circular buffer is full (pointer just wrapped to 0)
            # At this point, the buffer contains exactly local_window tokens that need archiving.
            # The buffer holds tokens at positions [kv_offset - local_window : kv_offset].
            # Future writes will overwrite these, so we archive the entire buffer now.
            if self.local_window_head < old_head:  # Wrapped around (buffer is full)
                # Archive all local_window tokens currently in the buffer
                # Buffer positions [0 : local_window] map to sequence positions [kv_offset - local_window : kv_offset]
                archive_start = self.kv_offset - self.local_window  # First sequence position in buffer
                archive_end = self.kv_offset  # One past last sequence position in buffer

                # Only archive positions we haven't archived yet
                start_pos = max(self.last_projected_pos, archive_start)
                end_pos = archive_end

                if start_pos < end_pos:
                    num_to_archive = end_pos - start_pos
                    # Offset within the buffer: buffer is linear [0:local_window] mapping to [archive_start:archive_end]
                    offset_in_buffer = start_pos - archive_start

                    # Archive keys and values for all layers
                    for archive_layer_idx in range(self.num_hidden_layers):
                        # Extract the slice to archive from circular buffer
                        keys_to_archive = self.k_cache_buffer[archive_layer_idx, :, :, offset_in_buffer:offset_in_buffer + num_to_archive]
                        values_to_archive = self.v_cache_buffer[archive_layer_idx, :, :, offset_in_buffer:offset_in_buffer + num_to_archive]

                        # Store to main cache
                        self.k_cache[archive_layer_idx, :, :, start_pos:end_pos].copy_(keys_to_archive)
                        self.v_cache[archive_layer_idx, :, :, start_pos:end_pos].copy_(values_to_archive)

                    # OPTIMIZATION: Batch project RoPEd keys for ALL layers at once
                    # Instead of 32 separate einsum ops, do 1 batched einsum
                    # Extract keys for all layers: [num_layers, B, num_kv_heads, num_to_archive, head_dim]
                    all_keys_to_archive = self.k_cache_buffer[:, :, :, offset_in_buffer:offset_in_buffer + num_to_archive]

                    # Batched einsum across all layers
                    # all_keys: [L, B, H, T, K] x proj_weights: [L, H, K, D] -> [L, B, H, T, D]
                    all_k_proj = torch.einsum("lbhtk,lhkd->lbhtd", all_keys_to_archive, self.predictor.key_cache_proj)

                    # Store all projections in one batched copy
                    self.k_proj_cache[:, :, :, start_pos:end_pos].copy_(all_k_proj)

                    # Update tracking - all buffer contents are now archived
                    self.last_projected_pos = end_pos
    
    def prefetch_layer_group(
        self,
        hidden_states: torch.Tensor,
        start_layer_idx: int,
    ):
        """
        Compute importance queries and perform batched retrieval for a group of layers.
        Called at producer layers (every producer_frequency layers).
        
        Args:
            hidden_states: [bsz, seq_len, hidden_size] - hidden states (can be 1 for decode or longer)
            start_layer_idx: Start layer index of the group
        """
        # Only use the last token's hidden state for importance computation
        if hidden_states.shape[1] > 1:
            hidden_states = hidden_states[:, -1:, :]
            
        bsz = hidden_states.shape[0]
        
        # 1. Predict Importance Queries
        with self._record('predictor_forward'):
            q_importance = self.predictor(hidden_states, producer_layer_idx=start_layer_idx)
        
        # 2. Batched Scoring & Selection & Retrieval
        producer_group_size = self.producer_frequency
        end_layer_idx = min(start_layer_idx + producer_group_size, self.num_hidden_layers)
        num_layers_in_group = end_layer_idx - start_layer_idx
        
        # Reshape and prepare tensors for batched operations
        with self._record('prepare_tensors'):
            # q_importance: [B*num_attn_heads, N_slots, 1, dDash] -> [B, num_attn_heads, N_slots, 1, dDash]
            q_importance = q_importance.view(bsz, self.num_attention_heads, self.producer_frequency, 1, self.dDash)
            q_group = q_importance[:, :, :num_layers_in_group, 0, :]  # [B, num_attn_heads, num_layers, dDash]
            q_group = q_group.permute(0, 2, 1, 3).unsqueeze(3)  # [B, num_layers, num_attn_heads, 1, dDash]
            
            # Get Projected Keys for these layers
            k_proj_group = self.k_proj_cache[start_layer_idx:end_layer_idx, :, :, :self.last_projected_pos, :]
            k_proj_group = k_proj_group.permute(1, 0, 2, 3, 4).unsqueeze(3)  # [B, num_layers, num_kv_heads, 1, limit, dDash]
            
            # Broadcast q_group to match KV heads (GQA)
            q_group = q_group.view(bsz, num_layers_in_group, self.num_key_value_heads, self.num_key_value_groups, 1, self.dDash)
            k_proj_group = k_proj_group.unsqueeze(3)  # [B, num_layers, num_kv_heads, 1, 1, limit, dDash]
        
        # 3. Compute Scores (Batched)
        with self._record('compute_scores'):
            scores = torch.einsum("blhgqd,blhgqkd->blhgqk", q_group, k_proj_group)
            scores = scores.squeeze(4) / math.sqrt(self.dDash)  # [B, Layers, KvH, Grp, Limit]
            scores = scores.max(dim=3).values  # [B, Layers, KvH, Limit]
            
            # Mask local window
            local_start = max(0, self.kv_offset - self.local_window)
            if local_start < self.last_projected_pos:
                scores[:, :, :, local_start:] = float("-inf")
            
        # 4. Select Indices (Batched)
        num_available = min(local_start, self.last_projected_pos)
        num_to_select = min(self.sparse_budget, num_available)
        
        if num_to_select > 0:
            with self._record('topk_selection'):
                _, topk_indices = torch.topk(scores, k=num_to_select, dim=-1)  # [B, Layers, KvH, budget]
                topk_indices, _ = topk_indices.sort(dim=-1)
            
            # 5. Batched Retrieval (Gather)
            # Prepare indices: [Layers, B, KvH, budget, HeadDim]
            indices_perm = topk_indices.permute(1, 0, 2, 3)
            indices_expanded = indices_perm.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
            
            sparse_start = self.local_window
            k_source = self.k_cache[start_layer_idx:end_layer_idx]
            v_source = self.v_cache[start_layer_idx:end_layer_idx]
            
            # Gather K
            with self._record('get_key_cache_total'):
                out_k = self.k_cache_buffer[start_layer_idx:end_layer_idx, :, :, sparse_start:sparse_start+num_to_select]
                torch.gather(k_source, dim=3, index=indices_expanded, out=out_k)
            
            # Gather V
            with self._record('get_value_cache_total'):
                out_v = self.v_cache_buffer[start_layer_idx:end_layer_idx, :, :, sparse_start:sparse_start+num_to_select]
                torch.gather(v_source, dim=3, index=indices_expanded, out=out_v)
            
        else:
            pass  # Nothing to select, buffer already has local window (handled by update)

    def get_retrieval_position_ids(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        No-op for batched KeySifter.
        """
        return None
    
    def get_key_cache(
        self,
        layer_idx: int,
        position_ids: Optional[torch.Tensor] = None,
        rope_func: Callable = None,
        cos_sin_cache: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Return the pre-fetched key cache buffer.
        """
        # We assume local window + sparse budget are already populated in the buffer
        # Logic for how many tokens to return:
        # If we selected num_to_select tokens, total is local_window + num_to_select
        # We typically fill up to sparse_budget if available.
        
        num_available = min(max(0, self.kv_offset - self.local_window), self.last_projected_pos)
        num_selected = min(self.sparse_budget, num_available)
        total_len = self.local_window + num_selected
        
        return self.k_cache_buffer[layer_idx, :, :, :total_len]
    
    def get_value_cache(
        self,
        layer_idx: int,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return the pre-fetched value cache buffer.
        """
        num_available = min(max(0, self.kv_offset - self.local_window), self.last_projected_pos)
        num_selected = min(self.sparse_budget, num_available)
        total_len = self.local_window + num_selected
        
        return self.v_cache_buffer[layer_idx, :, :, :total_len]
