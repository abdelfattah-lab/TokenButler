################################################################################
#
# Oracle KV Cache for xKV (Benchmarking Upper Bound)
# 
# This module implements the OracleCache class that uses random token selection
# with minimal overhead to represent the theoretical upper bound of performance.
#
################################################################################

import torch
import torch.nn.functional as F
import math
from typing import Optional, Callable

class OracleCache:
    """
    Oracle KV Cache for benchmarking upper bound performance.
    
    This class mimics the interface of TokenButlerCache/ShadowKV but skips:
        - Key projection
        - Importance scoring
        - Predictor usage
        
    It selects random tokens to fill the sparse budget, representing an
    "ideal" selection mechanism with zero computational overhead (approximate).
    """
    
    def __init__(
        self,
        config: object,
        batch_size: int = 1,
        max_length: int = 32 * 1024,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        sparse_budget: int = 2048,
        chunk_size: int = 8,
        producer_frequency: int = 4, # Kept for API compatibility, unused
        random_indices: bool = True,
        page_size: int = 1,
        predict_interval: int = 1,
        enable_neighbor_fetch: bool = False,
    ) -> None:
        """
        Args:
            config: Model configuration
            batch_size: Batch size
            max_length: Maximum sequence length
            device: Device to use
            dtype: Data type
            sparse_budget: Number of tokens to select during sparse attention
            chunk_size: Chunk size for selection (for compatibility)
            producer_frequency: Unused
            random_indices: If True, select random indices. If False, select first k contiguous indices.
            page_size: Size of contiguous pages to select randomly.
            predict_interval: Refresh sparse selection every N decode steps (1 = every step).
            enable_neighbor_fetch: Unused (for API compatibility).
        """
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        
        self.num_hidden_layers = config.num_hidden_layers
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_attention_heads
        
        self.sparse_budget = sparse_budget
        self.chunk_size = chunk_size
        self.producer_frequency = producer_frequency # Unused
        self.random_indices = random_indices
        self.page_size = page_size
        self.predict_interval = predict_interval
        self._cached_position_ids = None  # Cached position IDs for predict_interval > 1
        
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
        self.local_window_head = 0  # Circular queue head pointer for local window
        
        self.copy_stream = torch.cuda.Stream()

    def print_stats(self):
        print(f"OracleCache | sparse_budget {self.sparse_budget} | random_indices {self.random_indices} | page_size {self.page_size} | cached {self.kv_offset}")

    def get_kv_len(self):
        return self.kv_offset
    
    def clear(self):
        """Reset cache state for new sequence."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.k_cache_buffer.zero_()
        self.v_cache_buffer.zero_()
        self.kv_offset = 0
        self.prefill_len = 0
        self.gen_offset = 0
        self.local_window_head = 0
    
    def H2D(self):
        """Host to device transfer (no-op if already on GPU)."""
        pass
    
    def get_svd(self, key_states: torch.Tensor, layer_idx: int, fake_svd: bool = False):
        """No-op for OracleCache."""
        pass
    
    def prefill_kv_cache(
        self,
        new_v_cache: torch.Tensor,
        layer_idx: int,
        key_states_roped: torch.Tensor,
        query: torch.Tensor = None,
    ):
        """
        Store prefill K/V cache.
        Supports chunked prefill by accumulating across multiple calls.
        """
        incoming_len = new_v_cache.shape[2]
        current_len = self.prefill_len  # Start position for this chunk
        new_total_len = current_len + incoming_len
        
        # Store full RoPEd key cache at the correct position (for sparse retrieval during decode)
        self.k_cache[layer_idx, :, :, current_len:new_total_len].copy_(key_states_roped)
        
        # Store value cache
        self.v_cache[layer_idx, :, :, current_len:new_total_len].copy_(new_v_cache)
        
        # Initialize circular buffer with last local_window tokens from current chunk
        local_start = max(0, incoming_len - self.local_window)
        local_len = incoming_len - local_start
        self.k_cache_buffer[layer_idx, :, :, :local_len].copy_(
            key_states_roped[:, :, local_start:]
        )
        self.v_cache_buffer[layer_idx, :, :, :local_len].copy_(
            new_v_cache[:, :, local_start:]
        )
        
        if layer_idx == self.num_hidden_layers - 1:
            self.prefill_len = new_total_len
            self.kv_offset = new_total_len
            self.gen_offset = 0
    
    def update_kv_cache(
        self,
        new_k_cache: torch.Tensor,
        new_v_cache: torch.Tensor,
        layer_idx: int,
    ):
        """
        Update cache with new decode tokens.
        """
        incoming = new_k_cache.shape[2]
        if self.kv_offset + incoming > self.max_length:
            raise RuntimeError(
                f"Sequence length {self.kv_offset + incoming} exceeds max_length {self.max_length}"
            )

        # Add new token to circular buffer
        buffer_pos = self.local_window_head
        self.k_cache_buffer[layer_idx, :, :, buffer_pos:buffer_pos + incoming].copy_(new_k_cache)
        self.v_cache_buffer[layer_idx, :, :, buffer_pos:buffer_pos + incoming].copy_(new_v_cache)

        if layer_idx == self.num_hidden_layers - 1:
            # Update circular buffer head
            old_head = self.local_window_head
            self.local_window_head = (self.local_window_head + incoming) % self.local_window

            # Update position tracking
            self.kv_offset += incoming
            self.gen_offset += incoming

            # Archive if wrapped around
            if self.local_window_head < old_head:
                archive_start = self.kv_offset - self.local_window
                
                # Let's stick to the buffer archiving logic.
                # We need to copy from buffer to k_cache/v_cache.
                
                # Let's implement a 'last_archived_pos' to track what's in k_cache.
                if not hasattr(self, 'last_archived_pos'):
                    self.last_archived_pos = self.prefill_len
                
                start_pos = self.last_archived_pos
                end_pos = self.kv_offset
                
                if start_pos < end_pos:
                    # We have new tokens to archive.
                    start_pos = max(self.last_archived_pos, archive_start)
                    
                    if start_pos < end_pos:
                         for archive_layer_idx in range(self.num_hidden_layers):
                            # Copy all unarchived tokens from buffer to main cache
                            num_to_archive = end_pos - start_pos
                            offset_in_buffer = start_pos - archive_start
                            
                            keys_to_archive = self.k_cache_buffer[archive_layer_idx, :, :, offset_in_buffer:offset_in_buffer + num_to_archive]
                            values_to_archive = self.v_cache_buffer[archive_layer_idx, :, :, offset_in_buffer:offset_in_buffer + num_to_archive]
                            
                            self.k_cache[archive_layer_idx, :, :, start_pos:end_pos].copy_(keys_to_archive)
                            self.v_cache[archive_layer_idx, :, :, start_pos:end_pos].copy_(values_to_archive)
                            
                         self.last_archived_pos = end_pos

    
    def compute_predictor_importance(self, hidden_states: torch.Tensor, layer_idx: int):
        """No-op for OracleCache."""
        pass
    
    def prefetch_layer_group(
        self,
        hidden_states: torch.Tensor,
        start_layer_idx: int,
    ):
        """
        Pre-fill GPU buffer with sparse K/V for a group of layers.
        Respects predict_interval: on skipped steps, buffer is reused as-is.
        """
        # Prediction stride: skip on non-stride tokens, reuse previous buffer
        if self.predict_interval > 1 and self._cached_position_ids is not None:
            non_dense_offset = self.gen_offset - 1
            if non_dense_offset > 0 and non_dense_offset % self.predict_interval != 0:
                return

        # Generate new position IDs
        dummy_query = torch.zeros(
            self.batch_size, self.num_key_value_heads, 1, self.head_dim,
            device=self.device, dtype=self.dtype
        )
        position_ids = self._generate_position_ids(dummy_query)

        if self.predict_interval > 1:
            self._cached_position_ids = position_ids

        if position_ids.numel() == 0:
            return

        num_selected = position_ids.shape[-1]
        index = position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        sparse_start = self.local_window

        consumer_start = start_layer_idx
        consumer_end = min(start_layer_idx + self.producer_frequency, self.num_hidden_layers)

        for layer_idx in range(consumer_start, consumer_end):
            torch.gather(self.k_cache[layer_idx], dim=2, index=index,
                         out=self.k_cache_buffer[layer_idx, :, :, sparse_start:sparse_start + num_selected])
            torch.gather(self.v_cache[layer_idx], dim=2, index=index,
                         out=self.v_cache_buffer[layer_idx, :, :, sparse_start:sparse_start + num_selected])
    
    def get_retrieval_position_ids(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        """Return cached position IDs (buffer already filled by prefetch_layer_group)."""
        return self._cached_position_ids if self._cached_position_ids is not None else torch.zeros(
            query_states.shape[0], self.num_key_value_heads, 0, device=self.device, dtype=torch.long
        )

    def _generate_position_ids(self, query_states: torch.Tensor) -> torch.Tensor:
        """Generate fresh position IDs for sparse attention."""
        bsz = query_states.shape[0]

        if self.random_indices:
            # Random selection
            local_start = max(0, self.kv_offset - self.local_window)
            num_available = local_start # Can pick from [0, local_start)
            
            if self.page_size > 1:
                # Paged random selection
                num_pages_available = num_available // self.page_size
                num_pages_to_select = self.sparse_budget // self.page_size
                
                # Ensure we don't select more pages than available
                num_pages_to_select = min(num_pages_to_select, num_pages_available)
                
                if num_pages_to_select > 0:
                    # Pick random pages
                    random_page_indices = torch.randperm(num_pages_available, device=self.device)[:num_pages_to_select]
                    random_page_indices, _ = random_page_indices.sort()
                    
                    # Expand to full token indices
                    # [num_pages_to_select] -> [num_pages_to_select, page_size]
                    base_indices = random_page_indices * self.page_size
                    offsets = torch.arange(self.page_size, device=self.device)
                    
                    # [num_pages, 1] + [1, page_size] = [num_pages, page_size]
                    full_indices = base_indices.unsqueeze(1) + offsets.unsqueeze(0)
                    full_indices = full_indices.view(-1)
                    
                    # Expand to batch/heads like before
                    position_ids = full_indices.unsqueeze(0).unsqueeze(0).expand(bsz, self.num_key_value_heads, -1)
                else:
                    position_ids = torch.zeros(
                        bsz, self.num_key_value_heads, 0,
                        device=self.device, dtype=torch.long
                    )
            else:
                # Standard random selection
                num_to_select = min(self.sparse_budget, num_available)
                
                if num_to_select > 0:
                    # Generate random positions from [0, num_available) and sort them
                    random_positions = torch.randperm(num_available, device=self.device)[:num_to_select]
                    random_positions, _ = random_positions.sort()

                    # Expand to [bsz, num_kv_heads, num_to_select]
                    position_ids = random_positions.unsqueeze(0).unsqueeze(0).expand(bsz, self.num_key_value_heads, -1)
                else:
                    position_ids = torch.zeros(
                        bsz, self.num_key_value_heads, 0,
                        device=self.device, dtype=torch.long
                    )
        else:
            # Contiguous First-K selection
            # Just select [0, 1, ..., sparse_budget-1]
            position_ids = torch.arange(
                self.sparse_budget, device=self.device, dtype=torch.long
            ).unsqueeze(0).unsqueeze(0).expand(bsz, self.num_key_value_heads, -1)

        return position_ids

    def get_key_cache(
        self,
        layer_idx: int,
        position_ids: torch.Tensor,
        rope_func: Callable = None,
        cos_sin_cache: torch.Tensor = None,
    ) -> torch.Tensor:
        """Return key cache from pre-filled buffer."""
        num_selected = position_ids.shape[-1]
        total_len = self.local_window + num_selected
        return self.k_cache_buffer[layer_idx, :, :, :total_len]

    def get_value_cache(
        self,
        layer_idx: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return value cache from pre-filled buffer."""
        num_selected = position_ids.shape[-1]
        total_len = self.local_window + num_selected
        return self.v_cache_buffer[layer_idx, :, :, :total_len]
