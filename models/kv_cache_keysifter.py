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
        - Store full K/V cache
        - Project keys to reduced dimension (k_proj_cache) for efficient importance computation
    
    During decode:
        - Use predictor to compute importance queries
        - Score against projected key cache (avoids recomputation)
        - Select top-k positions based on importance scores
        - Retrieve and RoPE the selected keys for attention
    
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
        
        # Full key cache (un-RoPEd for SVD or direct access)
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
        
        # Buffer for selected K/V during decode
        buffer_size = sparse_budget + 4096  # Extra space for local window + new tokens
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
        self.q_importance_cache = None
    
    def H2D(self):
        """Host to device transfer (no-op if already on GPU)."""
        pass
    
    def get_svd(self, key_states: torch.Tensor, layer_idx: int, fake_svd: bool = False):
        """
        Bad name. Likely the old code needed to compute SVD. We're using it to store un-RoPEd keys and compute projections of keys for importance scoring.
        Called during prefill to prepare for sparse decode.
        For KeySifter, we also project those keys to be used during decode.
        
        Args:
            key_states: Either [bsz, seq_len, kv_dim] (flat) or [bsz, num_kv_heads, seq_len, head_dim]
            layer_idx: Layer index
            fake_svd: Compatibility flag (unused)
        """
        # Handle different input shapes (same logic as ShadowKVCache)
        if key_states.dim() == 3:
            # [bsz, seq_len, kv_dim] -> [bsz, num_kv_heads, seq_len, head_dim]
            bsz, seq_len, _ = key_states.shape
            key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        elif key_states.shape[1] > 32:
            # TODO: I think we can merge this with the first case?
            # [bsz, seq_len, kv_dim] format (seq_len > 32)
            bsz, seq_len, _ = key_states.shape
            key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # else: already [bsz, num_kv_heads, seq_len, head_dim]
        
        seq_len = key_states.shape[2]
        
        # Store un-RoPEd keys
        # TODO: Don't store here and store it in prefill_kv_cache instead so that it's RoPEd and we don't have to RoPE again during decode?
        self.k_cache[layer_idx, :, :, :seq_len].copy_(key_states)
        
        # Project keys for importance scoring (per KV head)
        # key_states: [B, num_kv_heads, L, head_dim]
        # proj_weight: [num_kv_heads, head_dim, dDash] (aggregated during predictor loading)
        proj_weight = self.predictor.key_cache_proj[layer_idx]  # [8, 128, dDash]
        k_proj = torch.einsum("bhlk,hkd->bhld", key_states, proj_weight)
        self.k_proj_cache[layer_idx, :, :, :seq_len].copy_(k_proj)
    
    def prefill_kv_cache(
        self,
        new_v_cache: torch.Tensor,
        layer_idx: int,
        key_states_roped: torch.Tensor,
        query: torch.Tensor = None,
    ):
        """
        Store prefill K/V cache and prepare for sparse decode.
        Note: The stored K is only the local window. 
        
        Args:
            new_v_cache: [bsz, num_kv_heads, seq_len, head_dim] - value states
            layer_idx: Layer index
            key_states_roped: [bsz, num_kv_heads, seq_len, head_dim] - RoPEd key states
            query: Optional query states (unused)
        """
        seq_len = new_v_cache.shape[2]
        
        # Store value cache
        self.v_cache[layer_idx, :, :, :seq_len].copy_(new_v_cache)
        
        # Store local window in buffer (always accessible during decode)
        local_start = max(0, seq_len - self.local_window)
        local_len = seq_len - local_start
        self.k_cache_buffer[layer_idx, :, :, :local_len].copy_(
            key_states_roped[:, :, local_start:]
        )
        self.v_cache_buffer[layer_idx, :, :, :local_len].copy_(
            new_v_cache[:, :, local_start:]
        )
        
        # Only update prefill_len after the last layer to ensure all layers see prefill_len=0
        # during the prefill pass (otherwise subsequent layers think prefill is complete)
        if layer_idx == self.num_hidden_layers - 1:
            self.prefill_len = seq_len
            self.kv_offset = seq_len
            self.gen_offset = 0
    
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
        
        # Compute buffer position (after local window + sparse budget)
        buffer_offset = self.local_window + self.sparse_budget + self.gen_offset
        
        # Add to buffer
        self.k_cache_buffer[layer_idx, :, :, buffer_offset:buffer_offset + incoming].copy_(new_k_cache)
        self.v_cache_buffer[layer_idx, :, :, buffer_offset:buffer_offset + incoming].copy_(new_v_cache)
        
        # TODO: Should implement the logic that would project and store un-RoPEd keys somewhere for importance scoring during long decode.
        # Note: We store the RoPEd key here; for proper importance scoring, we might
        # need to track position IDs separately. For now, decode tokens are in local window.
        
        if layer_idx == self.num_hidden_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming
    
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
        
        # Get projected keys up to prefill length (stored per KV head)
        # k_proj: [B, num_key_value_heads, prefill_len, dDash]
        k_proj = self.k_proj_cache[layer_idx, :, :, :self.prefill_len]
        
        # Reshape q_slot: [B*num_attention_heads, Lq, dDash] -> [B, num_attention_heads, Lq, dDash]
        Lq = q_slot.shape[1]
        q_slot = q_slot.view(bsz, self.num_attention_heads, Lq, self.dDash)
        
        # Use only the last query position for scoring
        q_slot = q_slot[:, :, -1:, :]  # [B, num_attention_heads, 1, dDash]
        
        # Efficient scoring with broadcasting:
        # q_slot: [B, num_attention_heads, 1, dDash] -> reshape to [B, num_kv_heads, num_kv_groups, 1, dDash]
        # k_proj: [B, num_key_value_heads, prefill_len, dDash] -> [B, num_kv_heads, 1, prefill_len, dDash]
        # This broadcasts k_proj across the num_kv_groups dimension efficiently
        q_slot = q_slot.view(bsz, self.num_key_value_heads, self.num_key_value_groups, 1, self.dDash)
        k_proj = k_proj.unsqueeze(2)  # [B, num_kv_heads, 1, prefill_len, dDash]
        
        # Compute scores: [B, num_kv_heads, num_kv_groups, 1, prefill_len]
        scores = torch.einsum("bhgqd,bhgkd->bhgqk", q_slot, k_proj)
        scores = scores.squeeze(3) / math.sqrt(self.dDash)  # [B, num_kv_heads, num_kv_groups, prefill_len]
        
        # Aggregate scores from attention heads to KV heads
        # Use max across the group (if any attention head thinks a token is important, keep it)
        scores = scores.max(dim=2).values  # [B, num_kv_heads, prefill_len]
        
        # Don't include local window positions (they're always included)
        local_start = max(0, self.prefill_len - self.local_window)
        scores[:, :, local_start:] = float("-inf")
        
        # Select top-k positions
        num_to_select = min(self.sparse_budget, local_start)
        if num_to_select > 0:
            _, position_ids = torch.topk(scores, k=num_to_select, dim=-1)  # [B, num_kv_heads, sparse_budget]
            # Sort positions for better memory access
            position_ids, _ = position_ids.sort(dim=-1)
        else:
            position_ids = torch.zeros(
                bsz, self.num_key_value_heads, 0,
                device=self.device, dtype=torch.long
            )
        
        return position_ids
    
    def get_key_cache(
        self,
        layer_idx: int,
        position_ids: torch.Tensor,
        rope_func: Callable,
        cos_sin_cache: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Gather selected keys from cache and apply RoPE.
        
        Args:
            layer_idx: Layer index
            position_ids: [bsz, num_kv_heads, sparse_budget] - positions to retrieve
            rope_func: Function to apply rotary position embeddings
            cos_sin_cache: Optional precomputed cos/sin cache
            
        Returns:
            key_states: [bsz, num_kv_heads, total_len, head_dim] - selected + local + decode keys
        """
        bsz = position_ids.shape[0]
        num_selected = position_ids.shape[-1]
        
        # Gather selected keys (un-RoPEd)
        # k_cache: [num_layers, bsz, num_kv_heads, max_len, head_dim]
        index = position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        selected_keys = self.k_cache[layer_idx].gather(dim=2, index=index)
        
        # Apply RoPE to selected keys
        selected_keys = rope_func(selected_keys, position_ids)
        
        # Copy to buffer at sparse position
        sparse_start = self.local_window
        sparse_end = sparse_start + num_selected
        self.k_cache_buffer[layer_idx, :, :, sparse_start:sparse_end].copy_(selected_keys)
        
        # Return full buffer (local + sparse + decode)
        total_len = self.local_window + num_selected + self.gen_offset
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
            value_states: [bsz, num_kv_heads, total_len, head_dim] - selected + local + decode values
        """
        num_selected = position_ids.shape[-1]
        
        # Gather selected values
        index = position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        selected_values = self.v_cache[layer_idx].gather(dim=2, index=index)
        
        # Copy to buffer at sparse position
        sparse_start = self.local_window
        sparse_end = sparse_start + num_selected
        self.v_cache_buffer[layer_idx, :, :, sparse_start:sparse_end].copy_(selected_values)
        
        # Return full buffer
        total_len = self.local_window + num_selected + self.gen_offset
        return self.v_cache_buffer[layer_idx, :, :, :total_len]
