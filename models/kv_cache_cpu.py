################################################################################
#
# Dense KV Cache with CPU Offloading
# 
# This module implements KV_Cache_CPU which stores the KV cache on CPU pinned
# memory and uses chunked streaming with online softmax for attention computation.
# This enables dense attention on contexts that exceed GPU memory.
#
################################################################################

import torch
import torch.nn.functional as F
import math
import gc
from typing import Optional, Tuple

from .kv_cache_cpu_base import CPUOffloadCacheBase


class KV_Cache_CPU(CPUOffloadCacheBase):
    """
    Dense KV Cache with CPU offloading using chunked streaming attention.
    
    Storage strategy:
    - k_cache, v_cache: CPU pinned memory (main storage)
    - Attention computed in chunks with online softmax accumulation
    
    During prefill:
        - Store K/V to CPU cache
        - Compute attention by streaming chunks to GPU
    
    During decode:
        - Stream KV cache in chunks from CPU to GPU
        - Use online softmax to accumulate attention output across chunks
        - Avoids materializing full attention matrix
    
    Online Softmax Algorithm:
        For each chunk i:
            1. Compute scores_i = Q @ K_i.T / sqrt(d)
            2. Compute local_max_i = max(scores_i)
            3. Update global_max: max_new = max(global_max, local_max_i)
            4. Rescale previous accumulator: scale = exp(global_max_old - max_new)
            5. Compute local softmax numerator: exp_scores_i = exp(scores_i - max_new)
            6. Update sum: sum_new = sum_old * scale + sum(exp_scores_i)
            7. Update output: output_new = output_old * scale + exp_scores_i @ V_i
        Final: output = output / sum
    """
    
    def __init__(
        self,
        config: object,
        batch_size: int = 1,
        max_length: int = 32 * 1024,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        chunk_size: int = 4096,
    ) -> None:
        """
        Args:
            config: Model configuration
            batch_size: Batch size
            max_length: Maximum sequence length
            device: GPU device for computation
            dtype: Data type
            chunk_size: Number of tokens per chunk for streaming attention
        """
        super().__init__(config, batch_size, max_length, device, dtype)
        
        self.chunk_size = chunk_size
        self.num_layers = config.num_hidden_layers
        
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
        
        # GPU buffers for streaming chunks (double-buffered for overlap)
        self.k_chunk_buffer = torch.zeros(
            2,  # Double buffer
            batch_size,
            self.num_key_value_heads,
            chunk_size,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        self.v_chunk_buffer = torch.zeros(
            2,  # Double buffer
            batch_size,
            self.num_key_value_heads,
            chunk_size,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        
        # Batch prefill tracking
        self.prefilled_batch = 0
        
        # Secondary stream for overlapped transfers
        self.transfer_stream = torch.cuda.Stream()
    
    def print_stats(self):
        print(f"KV_Cache_CPU | max_length {self.max_length} | chunk_size {self.chunk_size} | cached {self.kv_offset} | CPU offload enabled")
    
    def clear(self):
        """Reset cache state for new sequence."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.k_chunk_buffer.zero_()
        self.v_chunk_buffer.zero_()
        self.kv_offset = 0
        self.prefilled_batch = 0
    
    def H2D(self):
        """No-op - cache stays on CPU."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def update_kv_cache(
        self,
        new_k_cache: torch.Tensor,
        new_v_cache: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new tokens and return full KV for attention.
        
        For prefill (incoming > 1): Uses chunked streaming attention.
        For decode (incoming == 1): Uses chunked streaming attention.
        
        Args:
            new_k_cache: [bsz, num_kv_heads, incoming, head_dim] - new keys (GPU)
            new_v_cache: [bsz, num_kv_heads, incoming, head_dim] - new values (GPU)
            layer_idx: Layer index
            
        Returns:
            Tuple of (key, value) tensors for attention
        """
        bsz, _, incoming, _ = new_v_cache.shape
        
        if bsz == self.batch_size:
            self.prefilled_batch = 0
        
        # Store to CPU cache
        start_idx = self.prefilled_batch
        end_idx = self.prefilled_batch + bsz
        
        with torch.cuda.stream(self.copy_stream):
            self.k_cache[layer_idx, start_idx:end_idx, :, self.kv_offset:self.kv_offset + incoming].copy_(
                new_k_cache.cpu(), non_blocking=True
            )
            self.v_cache[layer_idx, start_idx:end_idx, :, self.kv_offset:self.kv_offset + incoming].copy_(
                new_v_cache.cpu(), non_blocking=True
            )
        
        # For the return value, we need the full cache up to current position
        # This will be used by the attention mechanism
        total_len = self.kv_offset + incoming
        
        # Update tracking at last layer
        if layer_idx == self.num_layers - 1:
            self.prefilled_batch += bsz
            if self.prefilled_batch == self.batch_size:
                self.kv_offset += incoming
        
        # Return slice indices instead of actual tensors
        # The actual chunked attention will be handled by compute_chunked_attention
        # For compatibility, we return markers that indicate chunked mode is needed
        return total_len, layer_idx
    
    def compute_chunked_attention(
        self,
        query: torch.Tensor,
        layer_idx: int,
        total_kv_len: int,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute attention using chunked streaming with online softmax.
        
        Args:
            query: [bsz, num_heads, seq_len, head_dim] - query states
            layer_idx: Layer index
            total_kv_len: Total KV cache length
            softmax_scale: Scaling factor (default: 1/sqrt(head_dim))
            
        Returns:
            Attention output [bsz, num_heads, seq_len, head_dim]
        """
        bsz, num_heads, seq_len, head_dim = query.shape
        
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)
        
        # Sync any pending CPU copies
        self.copy_stream.synchronize()
        
        # Initialize accumulators for online softmax
        # Using float32 for numerical stability
        output_acc = torch.zeros(
            bsz, num_heads, seq_len, head_dim,
            device=self.device, dtype=torch.float32
        )
        max_score = torch.full(
            (bsz, num_heads, seq_len, 1),
            float('-inf'),
            device=self.device, dtype=torch.float32
        )
        sum_exp = torch.zeros(
            bsz, num_heads, seq_len, 1,
            device=self.device, dtype=torch.float32
        )
        
        # Expand query for GQA if needed
        # query: [bsz, num_attention_heads, seq_len, head_dim]
        # Need to handle GQA where num_attention_heads > num_key_value_heads
        
        # Process KV cache in chunks
        num_chunks = (total_kv_len + self.chunk_size - 1) // self.chunk_size
        current_buffer = 0
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, total_kv_len)
            chunk_len = chunk_end - chunk_start
            
            # Start async transfer for next chunk (if exists)
            if chunk_idx + 1 < num_chunks:
                next_buffer = 1 - current_buffer
                next_start = (chunk_idx + 1) * self.chunk_size
                next_end = min(next_start + self.chunk_size, total_kv_len)
                next_len = next_end - next_start
                
                with torch.cuda.stream(self.transfer_stream):
                    self.k_chunk_buffer[next_buffer, :, :, :next_len].copy_(
                        self.k_cache[layer_idx, :, :, next_start:next_end].to(self.device),
                        non_blocking=True
                    )
                    self.v_chunk_buffer[next_buffer, :, :, :next_len].copy_(
                        self.v_cache[layer_idx, :, :, next_start:next_end].to(self.device),
                        non_blocking=True
                    )
            
            # Load current chunk (first chunk needs sync load)
            if chunk_idx == 0:
                self.k_chunk_buffer[current_buffer, :, :, :chunk_len].copy_(
                    self.k_cache[layer_idx, :, :, chunk_start:chunk_end].to(self.device)
                )
                self.v_chunk_buffer[current_buffer, :, :, :chunk_len].copy_(
                    self.v_cache[layer_idx, :, :, chunk_start:chunk_end].to(self.device)
                )
            
            # Get current chunk KV
            k_chunk = self.k_chunk_buffer[current_buffer, :, :, :chunk_len]  # [bsz, kv_heads, chunk_len, head_dim]
            v_chunk = self.v_chunk_buffer[current_buffer, :, :, :chunk_len]
            
            # Expand K/V for GQA: repeat each KV head for its group of attention heads
            # k_chunk: [bsz, num_kv_heads, chunk_len, head_dim]
            # -> [bsz, num_attention_heads, chunk_len, head_dim]
            if self.num_key_value_groups > 1:
                k_chunk = k_chunk.repeat_interleave(self.num_key_value_groups, dim=1)
                v_chunk = v_chunk.repeat_interleave(self.num_key_value_groups, dim=1)
            
            # Compute attention scores for this chunk
            # query: [bsz, num_heads, seq_len, head_dim]
            # k_chunk: [bsz, num_heads, chunk_len, head_dim]
            scores = torch.matmul(query, k_chunk.transpose(-2, -1)) * softmax_scale
            # scores: [bsz, num_heads, seq_len, chunk_len]
            
            # Online softmax update
            # Step 1: Get local max
            chunk_max = scores.max(dim=-1, keepdim=True).values  # [bsz, num_heads, seq_len, 1]
            
            # Step 2: Update global max
            new_max = torch.maximum(max_score, chunk_max)
            
            # Step 3: Rescale previous accumulator
            # exp(old_max - new_max) is the correction factor
            old_scale = torch.exp(max_score - new_max)
            output_acc = output_acc * old_scale
            sum_exp = sum_exp * old_scale
            
            # Step 4: Compute exp(scores - new_max)
            exp_scores = torch.exp(scores - new_max)  # [bsz, num_heads, seq_len, chunk_len]
            
            # Step 5: Update sum
            sum_exp = sum_exp + exp_scores.sum(dim=-1, keepdim=True)
            
            # Step 6: Update output accumulator
            # exp_scores @ v_chunk: [bsz, num_heads, seq_len, head_dim]
            output_acc = output_acc + torch.matmul(exp_scores.to(v_chunk.dtype), v_chunk).float()
            
            # Step 7: Update max for next iteration
            max_score = new_max
            
            # Wait for next chunk transfer before switching buffers
            if chunk_idx + 1 < num_chunks:
                self.transfer_stream.synchronize()
                current_buffer = 1 - current_buffer
        
        # Final normalization
        output = output_acc / sum_exp
        
        return output.to(self.dtype)
    
    def get_kv_for_prefill_attention(
        self,
        layer_idx: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get full KV cache for prefill attention (small enough to fit in GPU).
        
        For very long prefills, consider using compute_chunked_attention instead.
        
        Args:
            layer_idx: Layer index
            seq_len: Sequence length
            
        Returns:
            Tuple of (key, value) tensors on GPU
        """
        self.copy_stream.synchronize()
        
        k = self.k_cache[layer_idx, :, :, :seq_len].to(self.device)
        v = self.v_cache[layer_idx, :, :, :seq_len].to(self.device)
        
        return k, v
    
    # Legacy interface compatibility
    def get_svd(self, key_states: torch.Tensor, layer_idx: int, fake_svd: bool = False):
        """No-op for dense cache. Kept for API compatibility."""
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
        
        Args:
            new_v_cache: [bsz, num_kv_heads, seq_len, head_dim]
            layer_idx: Layer index
            key_states_roped: [bsz, num_kv_heads, seq_len, head_dim]
            query: Optional (unused)
        """
        seq_len = new_v_cache.shape[2]
        
        # Copy to CPU cache
        with torch.cuda.stream(self.copy_stream):
            self.k_cache[layer_idx, :, :, :seq_len].copy_(
                key_states_roped.cpu(), non_blocking=True
            )
            self.v_cache[layer_idx, :, :, :seq_len].copy_(
                new_v_cache.cpu(), non_blocking=True
            )
        
        if layer_idx == self.num_layers - 1:
            self.copy_stream.synchronize()
            self.kv_offset = seq_len
    
    def get_key_cache(
        self,
        layer_idx: int,
        position_ids: Optional[torch.Tensor] = None,
        rope_func=None,
        cos_sin_cache: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Get key cache - transfers from CPU to GPU.
        
        Note: For long contexts, use compute_chunked_attention instead.
        """
        self.copy_stream.synchronize()
        return self.k_cache[layer_idx, :, :, :self.kv_offset].to(self.device)
    
    def get_value_cache(
        self,
        layer_idx: int,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get value cache - transfers from CPU to GPU.
        
        Note: For long contexts, use compute_chunked_attention instead.
        """
        self.copy_stream.synchronize()
        return self.v_cache[layer_idx, :, :, :self.kv_offset].to(self.device)
