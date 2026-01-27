################################################################################
#
# CPU Offload Cache Base Class
# 
# This module provides a base class for KV caches that offload to CPU memory.
# Common functionality includes pinned memory allocation and async transfer streams.
#
################################################################################

import torch
import gc
from abc import ABC, abstractmethod
from typing import Optional


class CPUOffloadCacheBase(ABC):
    """
    Base class for KV caches that offload main cache storage to CPU.
    
    Provides common utilities:
    - Pinned memory allocation for efficient CPU-GPU transfers
    - CUDA stream for async transfers
    - Common interface methods (get_kv_len, clear, H2D, print_stats)
    """
    
    def __init__(
        self,
        config: object,
        batch_size: int = 1,
        max_length: int = 32 * 1024,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """
        Args:
            config: Model configuration object with num_hidden_layers, num_key_value_heads, etc.
            batch_size: Batch size
            max_length: Maximum sequence length
            device: GPU device for computation
            dtype: Data type for tensors
        """
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        
        self.num_hidden_layers = config.num_hidden_layers
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # KV offset tracking
        self.kv_offset = 0
        
        # CUDA stream for async CPU-GPU transfers
        self.copy_stream = torch.cuda.Stream()
        
    def _allocate_pinned_cache(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        max_length: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Allocate a pinned CPU tensor for KV cache storage.
        
        Pinned memory enables faster async CPU-GPU transfers.
        
        Args:
            num_layers: Number of transformer layers
            batch_size: Batch size
            num_heads: Number of KV heads
            max_length: Maximum sequence length
            head_dim: Head dimension
            dtype: Data type
            
        Returns:
            Pinned CPU tensor of shape [num_layers, batch_size, num_heads, max_length, head_dim]
        """
        return torch.zeros(
            num_layers,
            batch_size,
            num_heads,
            max_length,
            head_dim,
            device='cpu',
            dtype=dtype,
            pin_memory=True,
        )
    
    def _allocate_gpu_buffer(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        buffer_size: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Allocate a GPU buffer tensor for working set.
        
        Args:
            num_layers: Number of transformer layers
            batch_size: Batch size
            num_heads: Number of KV heads  
            buffer_size: Size of the buffer (tokens)
            head_dim: Head dimension
            dtype: Data type
            
        Returns:
            GPU tensor of shape [num_layers, batch_size, num_heads, buffer_size, head_dim]
        """
        return torch.zeros(
            num_layers,
            batch_size,
            num_heads,
            buffer_size,
            head_dim,
            device=self.device,
            dtype=dtype,
        )
    
    def get_kv_len(self) -> int:
        """Get current KV cache length (number of tokens stored)."""
        return self.kv_offset
    
    @abstractmethod
    def clear(self) -> None:
        """Reset cache state for new sequence."""
        pass
    
    @abstractmethod
    def print_stats(self) -> None:
        """Print cache statistics."""
        pass
    
    def H2D(self) -> None:
        """
        Host to device transfer hook.
        Called after prefill to transfer any needed data to GPU.
        Default implementation is a no-op; subclasses can override.
        """
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def _async_copy_to_gpu(
        self,
        src_cpu: torch.Tensor,
        dst_gpu: torch.Tensor,
        src_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Async copy from CPU to GPU using copy_stream.
        
        Args:
            src_cpu: Source tensor on CPU (pinned memory recommended)
            dst_gpu: Destination tensor on GPU
            src_indices: Optional indices for sparse gather (if None, copies entire tensor)
        """
        with torch.cuda.stream(self.copy_stream):
            if src_indices is None:
                dst_gpu.copy_(src_cpu, non_blocking=True)
            else:
                # Sparse gather - expand indices to match head_dim
                indices_expanded = src_indices.unsqueeze(-1).expand(-1, -1, -1, -1, dst_gpu.shape[-1])
                gathered = torch.gather(src_cpu, dim=3, index=indices_expanded)
                dst_gpu.copy_(gathered, non_blocking=True)
    
    def _sync_copy_stream(self) -> None:
        """Wait for async copy operations to complete."""
        self.copy_stream.synchronize()
