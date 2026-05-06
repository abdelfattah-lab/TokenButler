################################################################################
#
# TokenButler KV Cache for xKV
# 
# This module implements the TokenButlerCache class that uses a learned predictor
# for importance-based sparse attention during decoding.
#
################################################################################

import torch
import math
from typing import Optional, Callable

from .tokenbutler_predictor import TokenButlerPredictor

# Try to import fused INT8 kernel
try:
    from kernels.int8_score_fused import score_int8_fused
    HAS_INT8_FUSED_KERNEL = True
except ImportError:
    HAS_INT8_FUSED_KERNEL = False


class TokenButlerCache:
    """
    KV Cache with TokenButler-based sparse attention for decode.
    
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
        predictor: TokenButlerPredictor,
        batch_size: int = 1,
        max_length: int = 32 * 1024,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size: int = 8,
        producer_frequency: int = 4,
        local_window: int = 512,
        min_sparse_index: int = 128,
        quantize_int8: bool = False,
        predict_interval: int = 1,
        enable_neighbor_fetch: bool = False,
    ) -> None:
        """
        Args:
            config: Model configuration
            predictor: Trained TokenButlerPredictor instance
            batch_size: Batch size
            max_length: Maximum sequence length
            device: Device to use
            dtype: Data type
            sparse_budget: Number of tokens to select during sparse attention
            chunk_size: Chunk size for selection (for compatibility with ShadowKV interface)
            producer_frequency: Number of layers served by one predictor
            local_window: Number of recent tokens to always include (default 512, matching TokenButler baseline)
            min_sparse_index: Number of initial "sink" tokens to always keep (default 128, matching TokenButler baseline)
            quantize_int8: If True, store k_proj_cache as INT8 for memory bandwidth reduction
            predict_interval: Predict important tokens every N decode tokens (1 = every token, baseline)
            enable_neighbor_fetch: If True, double the sparse buffer and fetch neighbor tokens alongside important ones
        """
        self.config = config
        self.predictor = predictor
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.quantize_int8 = quantize_int8
        
        self.num_hidden_layers = config.num_hidden_layers
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_attention_heads
        
        self.sparse_budget = sparse_budget
        self.chunk_size = chunk_size
        self.producer_frequency = producer_frequency
        self.dDash = predictor.dDash
        self.predict_interval = predict_interval
        self.enable_neighbor_fetch = enable_neighbor_fetch
        self.effective_sparse_capacity = sparse_budget * 2 if enable_neighbor_fetch else sparse_budget

        # Local window to always include (matching TokenButler baseline's sliding_window)
        self.local_window = local_window  # Default 512 to match baseline

        # Sink tokens to always keep (matching TokenButler baseline's min_sparse_index)
        self.min_sparse_index = min_sparse_index  # Default 128 to match baseline
        
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
        if self.quantize_int8:
            # INT8 storage for memory bandwidth reduction
            self.k_proj_cache = torch.zeros(
                self.num_hidden_layers,
                batch_size,
                self.num_key_value_heads,
                max_length,
                self.dDash,
                device=device,
                dtype=torch.int8,
            )
            # Per-layer, per-head scale factors for dequantization
            # Shape: [num_layers, 1, num_key_value_heads, 1, 1]
            self.k_proj_scale = torch.ones(
                self.num_hidden_layers,
                1,
                self.num_key_value_heads,
                1,
                1,
                device=device,
                dtype=torch.float32,
            )
        else:
            self.k_proj_cache = torch.zeros(
                self.num_hidden_layers,
                batch_size,
                self.num_key_value_heads,
                max_length,
                self.dDash,
                device=device,
                dtype=dtype,
            )
            self.k_proj_scale = None
        
        # Buffer for assembling KV during decode: [sink_tokens | local_window (circular) | sparse_selected]
        # Layout: first min_sparse_index positions for sink tokens (always kept)
        #         next local_window positions for recent tokens (circular buffer)
        #         remaining positions for importance-selected tokens (2x when neighbor fetch is enabled)
        buffer_size = self.min_sparse_index + self.local_window + self.effective_sparse_capacity
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
        self._dense_decode_cutoff = 0  # Position after which to apply sparsity (keeps first decode token dense)
        self._last_num_sparse_selected = 0  # Actual number of sparse tokens in buffer (for neighbor fetch)
        self._force_next_prediction = False  # Force prediction on next decode step (used after prefill_cont)
        self.prefill_cont_dense = True  # If True, force predict_interval=1 during prefill_cont
        self._force_dense = False  # If True, get_key_cache/get_value_cache return full cache (dense attention)

        # Store importance queries from producer layers
        # Shape: [batch*heads, N_slots, 1, dDash] (for decode, Lq=1)
        self.q_importance_cache = None
        
        # Optional profiler for timing instrumentation (set externally)
        self.profiler = None

        self.copy_stream = torch.cuda.Stream()

        # During a single decode token, kv_offset is only "committed" at the last layer,
        # but caches for earlier layers already wrote the new token at position kv_offset.
        # We track an "uncommitted" increment so retrieval/selection uses the correct kv_len.
        self._uncommitted_decode = False
        self._uncommitted_incoming = 0

    def _kv_len_eff(self) -> int:
        """Effective KV length INCLUDING the current decode token (before last-layer commit)."""
        if self._uncommitted_decode and self._uncommitted_incoming > 0:
            return self.kv_offset + self._uncommitted_incoming
        return self.kv_offset

    def _record(self, name):
        """Context manager for profiling. No-op if profiler is None."""
        if self.profiler is not None:
            return self.profiler.record(name)
        else:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()
    
    def print_stats(self):
        print(f"TokenButlerCache | sparse_budget {self.sparse_budget} | producer_freq {self.producer_frequency} | dDash {self.dDash} | cached {self.kv_offset}")
    
    def get_kv_len(self):
        return self.kv_offset
    
    def clear(self):
        """Reset cache state for new sequence."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.k_proj_cache.zero_()
        if self.k_proj_scale is not None:
            self.k_proj_scale.fill_(1.0)
        self.k_cache_buffer.zero_()
        self.v_cache_buffer.zero_()
        self.kv_offset = 0
        self.prefill_len = 0
        self.gen_offset = 0
        self.last_projected_pos = 0
        self.local_window_head = 0
        self.q_importance_cache = None
        self._last_num_sparse_selected = 0
        self._force_dense = False
        self._uncommitted_decode = False
        self._uncommitted_incoming = 0

    def H2D(self):
        """Host to device transfer (no-op if already on GPU)."""
        pass
    
    def get_svd(self, key_states: torch.Tensor, layer_idx: int, fake_svd: bool = False):
        """
        No-op for TokenButlerCache. Kept for API compatibility with other cache implementations.
        
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
        Supports chunked prefill by accumulating across multiple calls.
        
        Args:
            new_v_cache: [bsz, num_kv_heads, seq_len, head_dim] - value states
            layer_idx: Layer index
            key_states_roped: [bsz, num_kv_heads, seq_len, head_dim] - RoPEd key states
            query: Optional query states (unused)
        """
        incoming_len = new_v_cache.shape[2]
        current_len = self.prefill_len  # Start position for this chunk
        new_total_len = current_len + incoming_len
        
        # Store full RoPEd key cache at the correct position (for sparse retrieval during decode)
        self.k_cache[layer_idx, :, :, current_len:new_total_len].copy_(key_states_roped)
        
        # Store value cache
        self.v_cache[layer_idx, :, :, current_len:new_total_len].copy_(new_v_cache)
        
        # Project RoPEd keys for importance scoring (per KV head)
        # key_states_roped: [B, num_kv_heads, L, head_dim]
        # proj_weight: [num_kv_heads, head_dim, dDash] (aggregated during predictor loading)
        proj_weight = self.predictor.key_cache_proj[layer_idx]  # [num_kv_heads, head_dim, dDash]
        k_proj = torch.einsum("bhlk,hkd->bhld", key_states_roped, proj_weight)

        if self.quantize_int8:
            # Quantize to INT8: compute per-head scale and store quantized values
            # k_proj: [B, num_kv_heads, L, dDash]
            # Compute scale per head (max abs value across positions and dDash dimensions)
            scale = k_proj.abs().amax(dim=(0, 2, 3), keepdim=True) / 127.0  # [1, num_kv_heads, 1, 1]
            scale = scale.clamp(min=1e-8)  # Avoid division by zero
            k_proj_int8 = (k_proj / scale).round().clamp(-128, 127).to(torch.int8)
            self.k_proj_cache[layer_idx, :, :, current_len:new_total_len].copy_(k_proj_int8)
            # Store scale - reshape to match [num_layers, 1, num_kv_heads, 1, 1]
            self.k_proj_scale[layer_idx].copy_(scale.squeeze(0).unsqueeze(0))
        else:
            self.k_proj_cache[layer_idx, :, :, current_len:new_total_len].copy_(k_proj)
        
        # Copy sink tokens to buffer (first min_sparse_index positions) - only on first chunk
        if current_len == 0:
            sink_len = min(self.min_sparse_index, incoming_len)
            if sink_len > 0:
                self.k_cache_buffer[layer_idx, :, :, :sink_len].copy_(
                    key_states_roped[:, :, :sink_len]
                )
                self.v_cache_buffer[layer_idx, :, :, :sink_len].copy_(
                    new_v_cache[:, :, :sink_len]
                )

        # Initialize circular buffer with last local_window tokens from current chunk
        # These will be the most recent tokens at the start of decode
        # Buffer layout: [sink_tokens | local_window | sparse_selected]
        local_buffer_start = self.min_sparse_index  # Offset for local window in buffer
        local_start = max(0, incoming_len - self.local_window)
        local_len = incoming_len - local_start
        self.k_cache_buffer[layer_idx, :, :, local_buffer_start:local_buffer_start + local_len].copy_(
            key_states_roped[:, :, local_start:]
        )
        self.v_cache_buffer[layer_idx, :, :, local_buffer_start:local_buffer_start + local_len].copy_(
            new_v_cache[:, :, local_start:]
        )
        # Circular buffer starts with head at 0, and local_len items filled
        # If incoming_len < local_window, only first local_len positions are valid
        
        # Only update prefill_len after the last layer to ensure all layers see same prefill_len
        # during the prefill pass (otherwise subsequent layers think prefill is complete)
        if layer_idx == self.num_hidden_layers - 1:
            self.prefill_len = new_total_len
            self.kv_offset = new_total_len
            self.gen_offset = 0
            self.last_projected_pos = new_total_len  # Prefill tokens are already projected
            self._dense_decode_cutoff = new_total_len + 1  # Keep first decode token dense (matching baseline)
            self._uncommitted_decode = False
            self._uncommitted_incoming = 0
    
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

        # Store decode token to main cache immediately (needed for layer 0 dense attention)
        # This ensures layers 0-3 (first producer group) can use full attention
        current_pos = self.kv_offset
        self.k_cache[layer_idx, :, :, current_pos:current_pos + incoming].copy_(new_k_cache)
        self.v_cache[layer_idx, :, :, current_pos:current_pos + incoming].copy_(new_v_cache)

        # Add new token to circular buffer
        # Buffer layout: [sink_tokens | local_window (circular) | sparse_selected]
        # Circular buffer starts at offset min_sparse_index
        local_buffer_start = self.min_sparse_index
        buffer_pos = local_buffer_start + self.local_window_head
        self.k_cache_buffer[layer_idx, :, :, buffer_pos:buffer_pos + incoming].copy_(new_k_cache)
        self.v_cache_buffer[layer_idx, :, :, buffer_pos:buffer_pos + incoming].copy_(new_v_cache)

        # Mark: there is an uncommitted token present at [kv_offset : kv_offset+incoming)
        # until the last layer increments kv_offset.
        if layer_idx != self.num_hidden_layers - 1:
            self._uncommitted_decode = True
            self._uncommitted_incoming = incoming

        if layer_idx == self.num_hidden_layers - 1:
            # Update circular buffer head (wraps around within local_window)
            old_head = self.local_window_head
            self.local_window_head = (self.local_window_head + incoming) % self.local_window

            # Update position tracking
            self.kv_offset += incoming
            self.gen_offset += incoming
            # Now the kv_offset is committed.
            self._uncommitted_decode = False
            self._uncommitted_incoming = 0
            self._force_next_prediction = False

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
                    # NOTE: local window region begins at min_sparse_index in k_cache_buffer
                    buf_start = local_buffer_start + offset_in_buffer
                    buf_end = buf_start + num_to_archive

                    # Archive keys and values for all layers
                    for archive_layer_idx in range(self.num_hidden_layers):
                        # Extract the slice to archive from circular buffer
                        keys_to_archive = self.k_cache_buffer[archive_layer_idx, :, :, buf_start:buf_end]
                        values_to_archive = self.v_cache_buffer[archive_layer_idx, :, :, buf_start:buf_end]

                        # Store to main cache
                        self.k_cache[archive_layer_idx, :, :, start_pos:end_pos].copy_(keys_to_archive)
                        self.v_cache[archive_layer_idx, :, :, start_pos:end_pos].copy_(values_to_archive)

                    # OPTIMIZATION: Batch project RoPEd keys for ALL layers at once
                    # Instead of 32 separate einsum ops, do 1 batched einsum
                    # Extract keys for all layers: [num_layers, B, num_kv_heads, num_to_archive, head_dim]
                    all_keys_to_archive = self.k_cache_buffer[:, :, :, buf_start:buf_end]

                    # Batched einsum across all layers
                    # all_keys: [L, B, H, T, K] x proj_weights: [L, H, K, D] -> [L, B, H, T, D]
                    all_k_proj = torch.einsum("lbhtk,lhkd->lbhtd", all_keys_to_archive, self.predictor.key_cache_proj)

                    if self.quantize_int8:
                        # Quantize to INT8 using existing per-layer, per-head scales
                        # all_k_proj: [L, B, H, T, D]
                        # k_proj_scale: [L, 1, H, 1, 1]
                        all_k_proj_int8 = (all_k_proj / self.k_proj_scale).round().clamp(-128, 127).to(torch.int8)
                        self.k_proj_cache[:, :, :, start_pos:end_pos].copy_(all_k_proj_int8)
                    else:
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
        # Skip sparse selection when forced dense (e.g., late turns in hybrid mode)
        if self._force_dense:
            return

        # Skip sparse selection for first decode token after prefill (matching baseline behavior)
        # Baseline keeps first decode token dense to avoid applying sparsity immediately after prefill
        kv_len = self._kv_len_eff()
        if kv_len <= self._dense_decode_cutoff:
            return  # Skip sparse selection, layers will use full cache via get_key_cache check

        # Prediction stride: skip prediction on non-stride tokens, reuse previous buffer.
        # gen_offset reflects completed decode tokens (committed at last layer).
        # The first decode token (gen_offset==0) is always dense (skipped above).
        # The first non-dense token (gen_offset==1) must always run prediction.
        # After that, predict every N tokens relative to the first non-dense token.
        if self.predict_interval > 1 and not self._force_next_prediction:
            non_dense_offset = self.gen_offset - 1  # 0-based from first non-dense token
            if non_dense_offset > 0 and non_dense_offset % self.predict_interval != 0:
                return

        # Only use the last token's hidden state for importance computation
        if hidden_states.shape[1] > 1:
            hidden_states = hidden_states[:, -1:, :]

        bsz = hidden_states.shape[0]
        
        # 1. Predict Importance Queries
        with self._record('predictor_forward'):
            q_importance = self.predictor(hidden_states, producer_layer_idx=start_layer_idx)

        # --- IMPORTANT (baseline parity) ---
        # Producer at layer p serves consumer layers (p+1 ... p+G), not (p ... p+G-1).
        consumer_start = start_layer_idx + 1
        if consumer_start >= self.num_hidden_layers:
            return
        consumer_end = min(consumer_start + self.producer_frequency, self.num_hidden_layers)
        num_layers_in_group = consumer_end - consumer_start
        if num_layers_in_group <= 0:
            return
        
        # Reshape and prepare tensors for batched operations
        with self._record('prepare_tensors'):
            # q_importance: [B*H, N_slots, 1, dDash] -> [B, H, N_slots, 1, dDash]
            q_importance = q_importance.view(bsz, self.num_attention_heads, self.producer_frequency, 1, self.dDash)
            # slots 0.. correspond to consumer layers p+1, p+2, ...
            q_slots = q_importance[:, :, :num_layers_in_group, 0, :]  # [B, H_attn, Lgrp, dDash]
            q_slots = q_slots.permute(0, 2, 1, 3)  # [B, Lgrp, H_attn, dDash]
            # Group by KV heads: [B, Lgrp, H_kv, G, dDash]
            q_group = q_slots.view(
                bsz,
                num_layers_in_group,
                self.num_key_value_heads,
                self.num_key_value_groups,
                self.dDash,
            )

            # Get projected keys for CONSUMER layers
            limit = self.last_projected_pos
            k_proj_group = self.k_proj_cache[consumer_start:consumer_end, :, :, :limit, :]  # [Lgrp, B, H_kv, limit, dDash]

        # 3. Compute Scores (Batched)
        with self._record('compute_scores'):
            if self.quantize_int8 and HAS_INT8_FUSED_KERNEL:
                # Use fused kernel: loads INT8, dequantizes on-the-fly, computes scores
                # This avoids creating a full bfloat16 copy of k_proj_group
                # q_group already: [B, Lgrp, H_kv, G, dDash]
                q_for_fused = q_group
                # k_proj_group is [L, B, H, Limit, D] int8
                # scale is [L, 1, H, 1, 1] float32
                scale_group = self.k_proj_scale[consumer_start:consumer_end]
                # Fused kernel returns [B, L, H, G, Limit]
                scores = score_int8_fused(q_for_fused, k_proj_group, scale_group)
                scores = scores / math.sqrt(self.dDash)  # [B, L, H, G, Limit]
                scores = scores.max(dim=3).values  # [B, L, H, Limit]
            else:
                # Non-INT8 path or fallback
                if self.quantize_int8:
                    # Fallback: Dequantize INT8 to float (less efficient, creates full copy)
                    scale_group = self.k_proj_scale[consumer_start:consumer_end].to(self.dtype)
                    k_proj_group = k_proj_group.to(self.dtype) * scale_group
                # [Lgrp, B, H_kv, limit, dDash] -> [B, Lgrp, H_kv, limit, dDash]
                k_proj_f = k_proj_group.permute(1, 0, 2, 3, 4)
                # scores: [B, Lgrp, H_kv, G, limit]
                scores = torch.einsum("blhgd,blhkd->blhgk", q_group, k_proj_f)
                scores = scores / math.sqrt(self.dDash)
                scores = scores.max(dim=3).values  # [B, Lgrp, H_kv, limit]

            # Mask sink tokens (always kept)
            limit = scores.size(-1)
            if self.min_sparse_index > 0:
                sink = min(self.min_sparse_index, limit)
                scores[:, :, :, :sink] = float("-inf")

            # Mask local window tokens (always kept) based on *effective* kv_len
            local_start = max(0, kv_len - self.local_window)
            if local_start < limit:
                scores[:, :, :, local_start:] = float("-inf")

            # Cast to the working dtype for top-k selection (no softmax needed
            # since we only care about relative ordering for top-k).
            scores = scores.to(self.dtype)
            
        # 4. Select Indices (Batched)
        # Available positions are between sink tokens and local window
        # Exclude: [0:min_sparse_index] (sink) and [local_start:] (local window)
        selection_end = min(local_start, self.last_projected_pos)
        selection_start = self.min_sparse_index
        num_available = max(0, selection_end - selection_start)
        num_to_select = min(self.sparse_budget, num_available)
        
        if num_to_select > 0:
            with self._record('topk_selection'):
                _, topk_indices = torch.topk(scores, k=num_to_select, dim=-1)  # [B, Layers, KvH, budget]
                topk_indices, _ = topk_indices.sort(dim=-1)

            # Expand with neighbors if enabled
            if self.enable_neighbor_fetch:
                topk_indices = self._expand_with_neighbors(
                    topk_indices, selection_start, selection_end
                )
                num_in_buffer = topk_indices.shape[-1]
            else:
                num_in_buffer = num_to_select

            self._last_num_sparse_selected = num_in_buffer

            # 5. Batched Retrieval (Gather)
            # Prepare indices: [Layers, B, KvH, num_in_buffer, HeadDim]
            indices_perm = topk_indices.permute(1, 0, 2, 3)
            indices_expanded = indices_perm.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)

            # Buffer layout: [sink_tokens | local_window | sparse_selected]
            # Sparse tokens go after sink_tokens and local_window
            sparse_start = self.min_sparse_index + self.local_window
            k_source = self.k_cache[consumer_start:consumer_end]
            v_source = self.v_cache[consumer_start:consumer_end]

            # Gather K
            with self._record('get_key_cache_total'):
                out_k = self.k_cache_buffer[consumer_start:consumer_end, :, :, sparse_start:sparse_start+num_in_buffer]
                torch.gather(k_source, dim=3, index=indices_expanded, out=out_k)

            # Gather V
            with self._record('get_value_cache_total'):
                out_v = self.v_cache_buffer[consumer_start:consumer_end, :, :, sparse_start:sparse_start+num_in_buffer]
                torch.gather(v_source, dim=3, index=indices_expanded, out=out_v)

        else:
            self._last_num_sparse_selected = 0

    def _expand_with_neighbors(
        self,
        topk_indices: torch.Tensor,
        selection_start: int,
        selection_end: int,
    ) -> torch.Tensor:
        """
        Expand top-k indices by adding unique neighboring tokens.
        Fully GPU-based using cluster-aware offsets.

        For each selected index, computes the size of its consecutive cluster
        and adds (index + cluster_size) as the neighbor. This ensures each
        element's neighbor lands just past the cluster end, producing unique
        entries without collisions within a cluster.

        Remaining gaps (from inter-cluster collisions or boundary clamping)
        are filled with sequential positions after the last valid entry.

        Target size: sparse_budget * 2.

        Args:
            topk_indices: [B, Layers, KvH, budget] sorted indices
            selection_start: Lower bound for valid indices (min_sparse_index)
            selection_end: Upper bound (exclusive) for valid indices

        Returns:
            Expanded sorted indices [B, Layers, KvH, target]
        """
        target = self.sparse_budget * 2
        B, L, H, K = topk_indices.shape
        device = topk_indices.device

        # Step 1: Compute cluster sizes for each element
        diffs = torch.zeros(B, L, H, K, device=device, dtype=topk_indices.dtype)
        diffs[..., 1:] = topk_indices[..., 1:] - topk_indices[..., :-1]
        is_cluster_start = (diffs != 1)  # position 0 has diff=0, so always True

        cluster_ids = is_cluster_start.long().cumsum(dim=-1) - 1
        num_clusters = int(cluster_ids.max().item()) + 1

        count_shape = list(cluster_ids.shape[:-1]) + [num_clusters]
        cluster_counts = torch.zeros(count_shape, device=device, dtype=topk_indices.dtype)
        cluster_counts.scatter_add_(-1, cluster_ids, torch.ones_like(topk_indices))

        elem_cluster_size = cluster_counts.gather(-1, cluster_ids)

        # Step 2: Neighbors skip past cluster end
        neighbors = (topk_indices + elem_cluster_size).clamp(max=selection_end - 1)

        # Step 3: Combine, sort, deduplicate
        combined = torch.cat([topk_indices, neighbors], dim=-1)
        combined_sorted, _ = combined.sort(dim=-1)

        dup_mask = torch.zeros_like(combined_sorted, dtype=torch.bool)
        dup_mask[..., 1:] = combined_sorted[..., 1:] == combined_sorted[..., :-1]
        combined_sorted[dup_mask] = selection_end
        combined_sorted, _ = combined_sorted.sort(dim=-1)

        # Step 4: Truncate or pad to target
        if combined_sorted.shape[-1] >= target:
            result = combined_sorted[..., :target]
        else:
            pad_val = combined_sorted[..., -1:]
            padding = pad_val.expand(*combined_sorted.shape[:-1], target - combined_sorted.shape[-1])
            result = torch.cat([combined_sorted, padding], dim=-1)

        # Step 5: Replace sentinels with sequential fill after last valid entry
        sentinel_mask = result >= selection_end
        if sentinel_mask.any():
            valid_vals = result.clone()
            valid_vals[sentinel_mask] = 0
            last_valid = valid_vals.max(dim=-1, keepdim=True).values

            sentinel_cumsum = sentinel_mask.long().cumsum(dim=-1)
            fill = (last_valid + sentinel_cumsum).clamp(max=selection_end - 1)
            result[sentinel_mask] = fill[sentinel_mask]

        return result

    def get_retrieval_position_ids(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        No-op for batched TokenButler.
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

        For layer 0 ONLY, return FULL cache to keep it dense.
        This matches TokenButler baseline behavior where only layer 0 uses dense attention.

        For first decode token after prefill, also return FULL cache (matching baseline).

        For other layers during normal decode:
        Buffer layout: [sink_tokens | local_window | sparse_selected]
        Total length: min_sparse_index + local_window + num_selected
        """
        # Layer 0 uses full attention - always return full cache
        # Also return full cache for first decode token after prefill (matching baseline)
        kv_len = self._kv_len_eff()
        if layer_idx == 0 or kv_len <= self._dense_decode_cutoff or self._force_dense:
            return self.k_cache[layer_idx, :, :, :kv_len]

        # Calculate how many sparse tokens were selected
        if self.enable_neighbor_fetch:
            num_selected = min(self._last_num_sparse_selected, self.effective_sparse_capacity)
        else:
            local_start = max(0, kv_len - self.local_window)
            selection_end = min(local_start, self.last_projected_pos)
            selection_start = self.min_sparse_index
            num_available = max(0, selection_end - selection_start)
            num_selected = min(self.sparse_budget, num_available)

        # Total tokens: sink + local_window + selected sparse
        total_len = self.min_sparse_index + self.local_window + num_selected

        return self.k_cache_buffer[layer_idx, :, :, :total_len]

    def get_value_cache(
        self,
        layer_idx: int,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return the pre-fetched value cache buffer.

        For layer 0 ONLY, return FULL cache to keep it dense.
        This matches TokenButler baseline behavior where only layer 0 uses dense attention.

        For first decode token after prefill, also return FULL cache (matching baseline).

        For other layers during normal decode:
        Buffer layout: [sink_tokens | local_window | sparse_selected]
        Total length: min_sparse_index + local_window + num_selected
        """
        # Layer 0 uses full attention - always return full cache
        # Also return full cache for first decode token after prefill (matching baseline)
        kv_len = self._kv_len_eff()
        if layer_idx == 0 or kv_len <= self._dense_decode_cutoff or self._force_dense:
            return self.v_cache[layer_idx, :, :, :kv_len]

        # Calculate how many sparse tokens were selected
        if self.enable_neighbor_fetch:
            num_selected = min(self._last_num_sparse_selected, self.effective_sparse_capacity)
        else:
            local_start = max(0, kv_len - self.local_window)
            selection_end = min(local_start, self.last_projected_pos)
            selection_start = self.min_sparse_index
            num_available = max(0, selection_end - selection_start)
            num_selected = min(self.sparse_budget, num_available)

        # Total tokens: sink + local_window + selected sparse
        total_len = self.min_sparse_index + self.local_window + num_selected

        return self.v_cache_buffer[layer_idx, :, :, :total_len]
