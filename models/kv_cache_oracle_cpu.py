################################################################################
#
# Oracle KV Cache with CPU Offloading
# 
# This module implements OracleCache_CPU which stores the main K/V cache
# on CPU pinned memory while keeping working buffers on GPU.
# Random/contiguous selection enables efficient CPU->GPU transfers for benchmarking.
#
################################################################################

import torch
import torch.nn.functional as F
import math
from typing import Optional, Callable

from .kv_cache_cpu_base import CPUOffloadCacheBase


class OracleCache_CPU(CPUOffloadCacheBase):
    """
    Oracle KV Cache for benchmarking upper bound performance with CPU offloading.
    
    Storage strategy:
    - k_cache, v_cache: CPU pinned memory (main storage for long contexts)
    - k_cache_buffer, v_cache_buffer: GPU (working set for attention)
    
    During prefill:
        - Store full RoPEd K/V to CPU cache
        - Copy sink tokens and local window to GPU buffer
    
    During decode:
        - Generate random or contiguous indices (oracle selection)
        - Gather only selected K/V from CPU to GPU buffer (sparse transfer)
        - Run attention on GPU buffer
        
    This class mimics the interface of KeySifterCache_CPU but skips:
        - Key projection
        - Importance scoring
        - Predictor usage
    
    It selects random/contiguous tokens to fill the sparse budget, representing
    an "ideal" selection mechanism with minimal computational overhead.
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
        producer_frequency: int = 4,  # Kept for API compatibility, unused
        local_window: int = 512,
        min_sparse_index: int = 128,
        random_indices: bool = True,
        page_size: int = 1,
    ) -> None:
        """
        Args:
            config: Model configuration
            batch_size: Batch size
            max_length: Maximum sequence length
            device: GPU device for computation
            dtype: Data type
            sparse_budget: Number of tokens to select during sparse attention
            chunk_size: Chunk size for selection (compatibility)
            producer_frequency: Unused (for API compatibility)
            local_window: Number of recent tokens to always include
            min_sparse_index: Number of initial "sink" tokens to always keep
            random_indices: If True, select random indices. If False, select first k contiguous indices.
            page_size: Size of contiguous pages to select randomly.
        """
        super().__init__(config, batch_size, max_length, device, dtype)
        
        self.sparse_budget = sparse_budget
        self.chunk_size = chunk_size
        self.producer_frequency = producer_frequency  # Unused
        self.local_window = local_window
        self.min_sparse_index = min_sparse_index
        self.random_indices = random_indices
        self.page_size = page_size
        
        # Main KV cache on CPU (pinned memory for fast transfers)
        self.k_cache = self._allocate_pinned_cache(
            self.num_hidden_layers,
            batch_size,
            self.num_key_value_heads,
            max_length,
            self.head_dim,
            dtype,
        )
        self.v_cache = self._allocate_pinned_cache(
            self.num_hidden_layers,
            batch_size,
            self.num_key_value_heads,
            max_length,
            self.head_dim,
            dtype,
        )
        
        # GPU working buffers: [sink_tokens | local_window | sparse_selected]
        buffer_size = self.min_sparse_index + self.local_window + sparse_budget
        self.k_cache_buffer = self._allocate_gpu_buffer(
            self.num_hidden_layers,
            batch_size,
            self.num_key_value_heads,
            buffer_size,
            self.head_dim,
            dtype,
        )
        self.v_cache_buffer = self._allocate_gpu_buffer(
            self.num_hidden_layers,
            batch_size,
            self.num_key_value_heads,
            buffer_size,
            self.head_dim,
            dtype,
        )
        
        # State tracking
        self.prefill_len = 0
        self.gen_offset = 0
        self.local_window_head = 0
        
    def print_stats(self):
        print(f"OracleCache_CPU | sparse_budget {self.sparse_budget} | random_indices {self.random_indices} | page_size {self.page_size} | cached {self.kv_offset} | CPU offload enabled")

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
        """No-op for OracleCache_CPU."""
        pass
    
    def prefill_kv_cache(
        self,
        new_v_cache: torch.Tensor,
        layer_idx: int,
        key_states_roped: torch.Tensor,
        query: torch.Tensor = None,
    ):
        """
        Store prefill K/V cache on CPU and prepare for sparse decode.
        
        Args:
            new_v_cache: [bsz, num_kv_heads, seq_len, head_dim] - value states (GPU)
            layer_idx: Layer index
            key_states_roped: [bsz, num_kv_heads, seq_len, head_dim] - RoPEd key states (GPU)
            query: Optional query states (unused)
        """
        seq_len = new_v_cache.shape[2]
        
        # Copy to CPU cache (async for efficiency)
        with torch.cuda.stream(self.copy_stream):
            self.k_cache[layer_idx, :, :, :seq_len].copy_(key_states_roped.cpu(), non_blocking=True)
            self.v_cache[layer_idx, :, :, :seq_len].copy_(new_v_cache.cpu(), non_blocking=True)
        
        # Copy sink tokens to GPU buffer (always kept)
        sink_len = min(self.min_sparse_index, seq_len)
        if sink_len > 0:
            self.k_cache_buffer[layer_idx, :, :, :sink_len].copy_(key_states_roped[:, :, :sink_len])
            self.v_cache_buffer[layer_idx, :, :, :sink_len].copy_(new_v_cache[:, :, :sink_len])
        
        # Initialize circular buffer with last local_window tokens
        local_buffer_start = self.min_sparse_index
        local_start = max(0, seq_len - self.local_window)
        local_len = seq_len - local_start
        self.k_cache_buffer[layer_idx, :, :, local_buffer_start:local_buffer_start + local_len].copy_(
            key_states_roped[:, :, local_start:]
        )
        self.v_cache_buffer[layer_idx, :, :, local_buffer_start:local_buffer_start + local_len].copy_(
            new_v_cache[:, :, local_start:]
        )
        
        if layer_idx == self.num_hidden_layers - 1:
            self.copy_stream.synchronize()  # Ensure CPU copies complete
            self.prefill_len = seq_len
            self.kv_offset = seq_len
            self.gen_offset = 0
            self.local_window_head = 0
    
    def update_kv_cache(
        self,
        new_k_cache: torch.Tensor,
        new_v_cache: torch.Tensor,
        layer_idx: int,
    ):
        """
        Update cache with new decode tokens.
        
        Args:
            new_k_cache: [bsz, num_kv_heads, 1, head_dim] - new RoPEd key (GPU)
            new_v_cache: [bsz, num_kv_heads, 1, head_dim] - new value (GPU)
            layer_idx: Layer index
        """
        incoming = new_k_cache.shape[2]
        
        if self.kv_offset + incoming > self.max_length:
            raise RuntimeError(
                f"Sequence length {self.kv_offset + incoming} exceeds max_length {self.max_length}"
            )
        
        current_pos = self.kv_offset
        
        # Store to CPU cache (async)
        with torch.cuda.stream(self.copy_stream):
            self.k_cache[layer_idx, :, :, current_pos:current_pos + incoming].copy_(
                new_k_cache.cpu(), non_blocking=True
            )
            self.v_cache[layer_idx, :, :, current_pos:current_pos + incoming].copy_(
                new_v_cache.cpu(), non_blocking=True
            )
        
        # Add new token to circular buffer (local window on GPU)
        local_buffer_start = self.min_sparse_index
        buffer_pos = local_buffer_start + self.local_window_head
        self.k_cache_buffer[layer_idx, :, :, buffer_pos:buffer_pos + incoming].copy_(new_k_cache)
        self.v_cache_buffer[layer_idx, :, :, buffer_pos:buffer_pos + incoming].copy_(new_v_cache)
        
        if layer_idx == self.num_hidden_layers - 1:
            self.copy_stream.synchronize()  # Ensure CPU copies complete
            
            # Update circular buffer head
            self.local_window_head = (self.local_window_head + incoming) % self.local_window
            
            # Update position tracking
            self.kv_offset += incoming
            self.gen_offset += incoming
    
    def compute_predictor_importance(self, hidden_states: torch.Tensor, layer_idx: int):
        """No-op for OracleCache_CPU."""
        pass
    
    def get_retrieval_position_ids(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get position IDs for sparse attention - Random or First-K selection for Oracle.
        
        Args:
            layer_idx: Layer index
            query_states: [bsz, num_kv_heads, 1, head_dim] - query states
            
        Returns:
            position_ids: [bsz, num_kv_heads, sparse_budget] - selected positions
        """
        bsz = query_states.shape[0]
        
        # Available positions for sparse selection (exclude sink + local window)
        local_start = max(0, self.kv_offset - self.local_window)
        num_available = max(0, local_start - self.min_sparse_index)
        
        if self.random_indices:
            # Random selection
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
                    # Offset by min_sparse_index to skip sink tokens
                    base_indices = random_page_indices * self.page_size + self.min_sparse_index
                    offsets = torch.arange(self.page_size, device=self.device)
                    
                    # [num_pages, 1] + [1, page_size] = [num_pages, page_size]
                    full_indices = base_indices.unsqueeze(1) + offsets.unsqueeze(0)
                    full_indices = full_indices.view(-1)
                    
                    # Expand to batch/heads
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
                    # Generate random positions from [min_sparse_index, local_start) and sort them
                    random_positions = torch.randperm(num_available, device=self.device)[:num_to_select] + self.min_sparse_index
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
            # Select [min_sparse_index, min_sparse_index + sparse_budget)
            num_to_select = min(self.sparse_budget, num_available)
            if num_to_select > 0:
                position_ids = torch.arange(
                    self.min_sparse_index,
                    self.min_sparse_index + num_to_select,
                    device=self.device, dtype=torch.long
                ).unsqueeze(0).unsqueeze(0).expand(bsz, self.num_key_value_heads, -1)
            else:
                position_ids = torch.zeros(
                    bsz, self.num_key_value_heads, 0,
                    device=self.device, dtype=torch.long
                )

        return position_ids
    
    def prefetch_layer_group(
        self,
        hidden_states: torch.Tensor,
        start_layer_idx: int,
    ):
        """
        Gather sparse K/V from CPU to GPU buffer for a group of layers.
        
        For Oracle, we generate random/contiguous indices and gather from CPU.
        Layer 0 is special - it gets full dense attention from CPU.
        
        Args:
            hidden_states: [bsz, 1, hidden_size] - current hidden states (unused for Oracle)
            start_layer_idx: Starting layer index for this producer group
        """
        if start_layer_idx == 0:
            # Layer 0: Full dense attention - return cache directly from CPU
            # Don't copy to GPU buffer since it's too small for full context
            # get_key_cache and get_value_cache will handle returning from CPU
            pass
        else:
            # Other layers: Sparse attention - generate indices and gather from CPU
            # Get oracle indices (same for all layers in group)
            dummy_query = torch.zeros(
                self.batch_size, self.num_key_value_heads, 1, self.head_dim,
                device=self.device, dtype=self.dtype
            )
            position_ids = self.get_retrieval_position_ids(start_layer_idx, dummy_query)
            
            if position_ids.numel() == 0:
                # No sparse positions to gather
                return
            
            num_selected = position_ids.shape[-1]
            sparse_buffer_start = self.min_sparse_index + self.local_window
            
            # Create gather indices for CPU tensors
            # position_ids: [bsz, num_kv_heads, num_selected]
            # Need: [bsz, num_kv_heads, num_selected, head_dim]
            indices_expanded = position_ids.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            
            # Move indices to CPU for gathering
            indices_cpu = indices_expanded.cpu()
            
            # Gather from CPU cache for all layers in the group
            consumer_start = start_layer_idx
            consumer_end = min(start_layer_idx + self.producer_frequency, self.num_hidden_layers)
            
            for layer_idx in range(consumer_start, consumer_end):
                # Gather keys
                k_gathered = torch.gather(self.k_cache[layer_idx], dim=2, index=indices_cpu)
                # Gather values
                v_gathered = torch.gather(self.v_cache[layer_idx], dim=2, index=indices_cpu)
                
                # Transfer gathered results to GPU buffer
                with torch.cuda.stream(self.copy_stream):
                    self.k_cache_buffer[layer_idx, :, :, sparse_buffer_start:sparse_buffer_start + num_selected].copy_(
                        k_gathered.to(self.device, non_blocking=True)
                    )
                    self.v_cache_buffer[layer_idx, :, :, sparse_buffer_start:sparse_buffer_start + num_selected].copy_(
                        v_gathered.to(self.device, non_blocking=True)
                    )
            
            self.copy_stream.synchronize()
    
    def get_key_cache(
        self,
        layer_idx: int,
        position_ids: torch.Tensor,
        rope_func: Callable = None,
        cos_sin_cache: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Get key cache for attention computation.
        
        For layer 0: Return full cache from GPU buffer
        For other layers: Return [sink | local_window | sparse_selected] from GPU buffer
        
        Args:
            layer_idx: Layer index
            position_ids: [bsz, num_kv_heads, sparse_budget] - selected positions (ignored, already in buffer)
            rope_func: RoPE function (unused, keys are already RoPEd)
            cos_sin_cache: RoPE cache (unused)
            
        Returns:
            key_cache: [bsz, num_kv_heads, total_len, head_dim]
        """
        if layer_idx == 0:
            # Layer 0: Return full cache from CPU (transfer to GPU on-the-fly)
            total_len = self.kv_offset
            return self.k_cache[layer_idx, :, :, :total_len].to(self.device)
        else:
            # Other layers: Return [sink | local_window | sparse] from GPU buffer
            num_selected = position_ids.shape[-1]
            total_len = self.min_sparse_index + self.local_window + num_selected
            return self.k_cache_buffer[layer_idx, :, :, :total_len]
    
    def get_value_cache(
        self,
        layer_idx: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get value cache for attention computation.
        
        For layer 0: Return full cache from GPU buffer
        For other layers: Return [sink | local_window | sparse_selected] from GPU buffer
        
        Args:
            layer_idx: Layer index
            position_ids: [bsz, num_kv_heads, sparse_budget] - selected positions (ignored, already in buffer)
            
        Returns:
            value_cache: [bsz, num_kv_heads, total_len, head_dim]
        """
        if layer_idx == 0:
            # Layer 0: Return full cache from CPU (transfer to GPU on-the-fly)
            total_len = self.kv_offset
            return self.v_cache[layer_idx, :, :, :total_len].to(self.device)
        else:
            # Other layers: Return [sink | local_window | sparse] from GPU buffer
            num_selected = position_ids.shape[-1]
            total_len = self.min_sparse_index + self.local_window + num_selected
            return self.v_cache_buffer[layer_idx, :, :, :total_len]
