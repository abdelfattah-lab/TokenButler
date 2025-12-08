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
        
        self.copy_stream = torch.cuda.Stream()
    
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
    
    def compute_predictor_importance(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ):
        """
        Compute importance queries using the predictor.
        Called at producer layers.
        
        Args:
            hidden_states: [bsz, seq_len, hidden_size] - hidden states (can be 1 for decode or longer)
            layer_idx: Layer index
        """
        # Only use the last token's hidden state for importance computation
        # This ensures consistent behavior during both prefill and decode
        if hidden_states.shape[1] > 1:
            hidden_states = hidden_states[:, -1:, :]
        
        # Compute importance queries
        # Output: [B*H, N_slots, 1, dDash]
        q_importance = self.predictor(hidden_states, producer_layer_idx=layer_idx)
        self.q_importance_cache = q_importance
    
    def get_retrieval_position_ids(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get position IDs for sparse attention based on importance scores.
        
        Args:
            layer_idx: Layer index
            query_states: [bsz, num_heads, 1, head_dim] - query states (for interface compatibility)
            
        Returns:
            position_ids: [bsz, num_kv_heads, sparse_budget] - selected positions
        """
        bsz = query_states.shape[0]
        
        # Determine which slot to use (within producer group)
        slot_idx = layer_idx % self.producer_frequency
        
        # Get importance queries for this slot
        if self.q_importance_cache is None:
            raise RuntimeError("Importance queries not computed. Call compute_predictor_importance first.")
        
        # q_importance: [B*num_attention_heads, N_slots, Lq, dDash] where Lq should be 1
        # (predictor uses num_attention_heads=32, not num_kv_heads=8)
        # Select slot: [B*num_attention_heads, Lq, dDash]
        q_slot = self.q_importance_cache[:, slot_idx, :, :]  # [B*32, Lq, dDash]
        
        # Get projected keys up to last projected position (includes prefill + archived decode tokens)
        # k_proj: [B, num_key_value_heads, last_projected_pos, dDash]
        k_proj = self.k_proj_cache[layer_idx, :, :, :self.last_projected_pos]
        
        # Reshape q_slot: [B*num_attention_heads, Lq, dDash] -> [B, num_attention_heads, Lq, dDash]
        Lq = q_slot.shape[1]
        q_slot = q_slot.view(bsz, self.num_attention_heads, Lq, self.dDash)
        
        # Use only the last query position for scoring
        q_slot = q_slot[:, :, -1:, :]  # [B, num_attention_heads, 1, dDash]
        
        # Efficient scoring with broadcasting:
        # q_slot: [B, num_attention_heads, 1, dDash] -> reshape to [B, num_kv_heads, num_kv_groups, 1, dDash]
        # k_proj: [B, num_key_value_heads, last_projected_pos, dDash] -> [B, num_kv_heads, 1, last_projected_pos, dDash]
        # This broadcasts k_proj across the num_kv_groups dimension efficiently
        q_slot = q_slot.view(bsz, self.num_key_value_heads, self.num_key_value_groups, 1, self.dDash)
        k_proj = k_proj.unsqueeze(2)  # [B, num_kv_heads, 1, last_projected_pos, dDash]

        # MODIFIED: Skip scoring and topk, use random positions instead
        # This is for benchmarking to measure topk overhead

        # # Original code (commented out):
        # Compute scores: [B, num_kv_heads, num_kv_groups, 1, last_projected_pos]
        scores = torch.einsum("bhgqd,bhgkd->bhgqk", q_slot, k_proj)
        scores = scores.squeeze(3) / math.sqrt(self.dDash)  # [B, num_kv_heads, num_kv_groups, last_projected_pos]
        
        # Aggregate scores from attention heads to KV heads
        # Use max across the group (if any attention head thinks a token is important, keep it)
        scores = scores.max(dim=2).values  # [B, num_kv_heads, last_projected_pos]
        
        # Don't include local window positions (they're always included)
        # Local window is the most recent tokens: [kv_offset - local_window : kv_offset]
        # We mask out any projected tokens that fall in this range
        local_start = max(0, self.kv_offset - self.local_window)
        if local_start < self.last_projected_pos:
            scores[:, :, local_start:] = float("-inf")
        
        # Select top-k positions from available tokens (projected tokens outside local window)
        num_available = min(local_start, self.last_projected_pos)
        num_to_select = min(self.sparse_budget, num_available)
        if num_to_select > 0:
            _, position_ids = torch.topk(scores, k=num_to_select, dim=-1)  # [B, num_kv_heads, sparse_budget]
            # Sort positions for better memory access
            position_ids, _ = position_ids.sort(dim=-1)
        else:
            position_ids = torch.zeros(
                bsz, self.num_key_value_heads, 0,
                device=self.device, dtype=torch.long
            )

        # # New code: Random selection
        # local_start = max(0, self.kv_offset - self.local_window)
        # num_available = min(local_start, self.last_projected_pos)
        # num_to_select = min(self.sparse_budget, num_available)

        # if num_to_select > 0:
        #     # Generate random positions from [0, num_available) and sort them
        #     # Use same positions for all batches and KV heads for simplicity
        #     random_positions = torch.randperm(num_available, device=self.device)[:num_to_select]
        #     random_positions, _ = random_positions.sort()

        #     # Expand to [bsz, num_kv_heads, num_to_select]
        #     position_ids = random_positions.unsqueeze(0).unsqueeze(0).expand(bsz, self.num_key_value_heads, -1)
        # else:
        #     position_ids = torch.zeros(
        #         bsz, self.num_key_value_heads, 0,
        #         device=self.device, dtype=torch.long
        #     )

        return position_ids
    
    def get_key_cache(
        self,
        layer_idx: int,
        position_ids: torch.Tensor,
        rope_func: Callable = None,
        cos_sin_cache: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Gather selected keys from cache (already RoPEd).

        Args:
            layer_idx: Layer index
            position_ids: [bsz, num_kv_heads, sparse_budget] - positions to retrieve
            rope_func: Unused (kept for interface compatibility). Keys are already RoPEd.
            cos_sin_cache: Unused (kept for interface compatibility)

        Returns:
            key_states: [bsz, num_kv_heads, total_len, head_dim] - local + sparse selected keys
        """
        num_selected = position_ids.shape[-1]

        # Gather selected keys from main cache (already RoPEd - stored during prefill or archived)
        # OPTIMIZATION: Use out= parameter to write directly to buffer, eliminating intermediate tensor
        index = position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        sparse_start = self.local_window
        torch.gather(
            self.k_cache[layer_idx],
            dim=2,
            index=index,
            out=self.k_cache_buffer[layer_idx, :, :, sparse_start:sparse_start + num_selected]
        )

        # Return buffer: [local_window (circular) | sparse_selected]
        # Order doesn't matter for flash attention as long as K and V are aligned
        total_len = self.local_window + num_selected
        return self.k_cache_buffer[layer_idx, :, :, :total_len]
    
    def get_value_cache(
        self,
        layer_idx: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather selected values from cache.

        Args:
            layer_idx: Layer index
            position_ids: [bsz, num_kv_heads, sparse_budget] - positions to retrieve

        Returns:
            value_states: [bsz, num_kv_heads, total_len, head_dim] - local + sparse selected values
        """
        num_selected = position_ids.shape[-1]

        # Gather selected values from main cache
        # OPTIMIZATION: Use out= parameter to write directly to buffer, eliminating intermediate tensor
        index = position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        sparse_start = self.local_window
        torch.gather(
            self.v_cache[layer_idx],
            dim=2,
            index=index,
            out=self.v_cache_buffer[layer_idx, :, :, sparse_start:sparse_start + num_selected]
        )

        # Return buffer: [local_window (circular) | sparse_selected]
        total_len = self.local_window + num_selected
        return self.v_cache_buffer[layer_idx, :, :, :total_len]
