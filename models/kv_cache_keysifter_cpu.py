################################################################################
#
# KeySifter KV Cache with CPU Offloading
# 
# This module implements KeySifterCache_CPU which stores the main K/V cache
# on CPU pinned memory while keeping working buffers on GPU.
# Sparse retrieval enables efficient CPU->GPU transfers for only selected tokens.
#
################################################################################

import torch
import torch.nn.functional as F
import math
from typing import Optional, Callable
from contextlib import nullcontext

from .kv_cache_cpu_base import CPUOffloadCacheBase
from .keysifter_predictor import KeySifterPredictor

# Try to import fused INT8 kernel
try:
    from kernels.int8_score_fused import score_int8_fused
    HAS_INT8_FUSED_KERNEL = True
except ImportError:
    HAS_INT8_FUSED_KERNEL = False


class KeySifterCache_CPU(CPUOffloadCacheBase):
    """
    KV Cache with KeySifter-based sparse attention and CPU offloading.
    
    Storage strategy:
    - k_cache, v_cache: CPU pinned memory (main storage for long contexts)
    - k_proj_cache: GPU (small, always needed for importance scoring)
    - k_cache_buffer, v_cache_buffer: GPU (working set for attention)
    
    During prefill:
        - Store full RoPEd K/V to CPU cache
        - Project keys to k_proj_cache on GPU for importance scoring
        - Copy sink tokens and local window to GPU buffer
    
    During decode:
        - Use predictor to compute importance scores against k_proj_cache (GPU)
        - Select top-k positions based on importance
        - Gather only selected K/V from CPU to GPU buffer (sparse transfer)
        - Run attention on GPU buffer
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
        local_window: int = 512,
        min_sparse_index: int = 128,
        quantize_int8: bool = False,
    ) -> None:
        """
        Args:
            config: Model configuration
            predictor: Trained KeySifterPredictor instance
            batch_size: Batch size
            max_length: Maximum sequence length
            device: GPU device for computation
            dtype: Data type
            sparse_budget: Number of tokens to select during sparse attention
            chunk_size: Chunk size for selection (compatibility with ShadowKV interface)
            producer_frequency: Number of layers served by one predictor
            local_window: Number of recent tokens to always include
            min_sparse_index: Number of initial "sink" tokens to always keep
            quantize_int8: If True, store k_proj_cache as INT8
        """
        super().__init__(config, batch_size, max_length, device, dtype)
        
        self.predictor = predictor
        self.quantize_int8 = quantize_int8
        
        self.sparse_budget = sparse_budget
        self.chunk_size = chunk_size
        self.producer_frequency = producer_frequency
        self.dDash = predictor.dDash
        self.local_window = local_window
        self.min_sparse_index = min_sparse_index
        
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
        
        # Projected key cache on GPU (small, always needed for importance scoring)
        if self.quantize_int8:
            self.k_proj_cache = torch.zeros(
                self.num_hidden_layers,
                batch_size,
                self.num_key_value_heads,
                max_length,
                self.dDash,
                device=device,
                dtype=torch.int8,
            )
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
        self.last_projected_pos = 0
        self.local_window_head = 0
        self._dense_decode_cutoff = 0
        
        self.q_importance_cache = None
        self.profiler = None
        
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
        return nullcontext()
    
    def print_stats(self):
        print(f"KeySifterCache_CPU | sparse_budget {self.sparse_budget} | producer_freq {self.producer_frequency} | dDash {self.dDash} | cached {self.kv_offset} | CPU offload enabled")
    
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
        self._uncommitted_decode = False
        self._uncommitted_incoming = 0
    
    def get_svd(self, key_states: torch.Tensor, layer_idx: int, fake_svd: bool = False):
        """No-op for KeySifterCache. Kept for API compatibility."""
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
        
        # Project RoPEd keys for importance scoring (keep on GPU)
        proj_weight = self.predictor.key_cache_proj[layer_idx]
        k_proj = torch.einsum("bhlk,hkd->bhld", key_states_roped, proj_weight)
        
        if self.quantize_int8:
            scale = k_proj.abs().amax(dim=(0, 2, 3), keepdim=True) / 127.0
            scale = scale.clamp(min=1e-8)
            k_proj_int8 = (k_proj / scale).round().clamp(-128, 127).to(torch.int8)
            self.k_proj_cache[layer_idx, :, :, :seq_len].copy_(k_proj_int8)
            self.k_proj_scale[layer_idx].copy_(scale.squeeze(0).unsqueeze(0))
        else:
            self.k_proj_cache[layer_idx, :, :, :seq_len].copy_(k_proj)
        
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
            self.last_projected_pos = seq_len
            self._dense_decode_cutoff = seq_len + 1
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
        
        # Add to GPU circular buffer
        local_buffer_start = self.min_sparse_index
        buffer_pos = local_buffer_start + self.local_window_head
        self.k_cache_buffer[layer_idx, :, :, buffer_pos:buffer_pos + incoming].copy_(new_k_cache)
        self.v_cache_buffer[layer_idx, :, :, buffer_pos:buffer_pos + incoming].copy_(new_v_cache)
        
        if layer_idx != self.num_hidden_layers - 1:
            self._uncommitted_decode = True
            self._uncommitted_incoming = incoming
        
        if layer_idx == self.num_hidden_layers - 1:
            old_head = self.local_window_head
            self.local_window_head = (self.local_window_head + incoming) % self.local_window
            
            self.kv_offset += incoming
            self.gen_offset += incoming
            self._uncommitted_decode = False
            self._uncommitted_incoming = 0
            
            # Archive when circular buffer wraps
            if self.local_window_head < old_head:
                archive_start = self.kv_offset - self.local_window
                archive_end = self.kv_offset
                
                start_pos = max(self.last_projected_pos, archive_start)
                end_pos = archive_end
                
                if start_pos < end_pos:
                    num_to_archive = end_pos - start_pos
                    offset_in_buffer = start_pos - archive_start
                    buf_start = local_buffer_start + offset_in_buffer
                    buf_end = buf_start + num_to_archive
                    
                    # Wait for any pending CPU copies
                    self.copy_stream.synchronize()
                    
                    # Project and store to k_proj_cache (batched across all layers)
                    all_keys = torch.zeros(
                        self.num_hidden_layers,
                        self.batch_size,
                        self.num_key_value_heads,
                        num_to_archive,
                        self.head_dim,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    all_keys.copy_(self.k_cache_buffer[:, :, :, buf_start:buf_end])
                    
                    all_k_proj = torch.einsum(
                        "lbhtk,lhkd->lbhtd",
                        all_keys,
                        self.predictor.key_cache_proj
                    )
                    
                    if self.quantize_int8:
                        all_k_proj_int8 = (all_k_proj / self.k_proj_scale).round().clamp(-128, 127).to(torch.int8)
                        self.k_proj_cache[:, :, :, start_pos:end_pos].copy_(all_k_proj_int8)
                    else:
                        self.k_proj_cache[:, :, :, start_pos:end_pos].copy_(all_k_proj)
                    
                    self.last_projected_pos = end_pos
    
    def prefetch_layer_group(
        self,
        hidden_states: torch.Tensor,
        start_layer_idx: int,
    ):
        """
        Compute importance queries and perform batched retrieval for a group of layers.
        Fetches selected K/V from CPU to GPU buffer.
        
        Args:
            hidden_states: [bsz, seq_len, hidden_size]
            start_layer_idx: Start layer index of the group
        """
        kv_len = self._kv_len_eff()
        if kv_len <= self._dense_decode_cutoff:
            return
        
        if hidden_states.shape[1] > 1:
            hidden_states = hidden_states[:, -1:, :]
        
        bsz = hidden_states.shape[0]
        
        # Predict importance queries
        with self._record('predictor_forward'):
            q_importance = self.predictor(hidden_states, producer_layer_idx=start_layer_idx)
        
        consumer_start = start_layer_idx + 1
        if consumer_start >= self.num_hidden_layers:
            return
        consumer_end = min(consumer_start + self.producer_frequency, self.num_hidden_layers)
        num_layers_in_group = consumer_end - consumer_start
        if num_layers_in_group <= 0:
            return
        
        # Prepare tensors
        with self._record('prepare_tensors'):
            q_importance = q_importance.view(bsz, self.num_attention_heads, self.producer_frequency, 1, self.dDash)
            q_slots = q_importance[:, :, :num_layers_in_group, 0, :]
            q_slots = q_slots.permute(0, 2, 1, 3)
            q_group = q_slots.view(
                bsz,
                num_layers_in_group,
                self.num_key_value_heads,
                self.num_key_value_groups,
                self.dDash,
            )
            
            limit = self.last_projected_pos
            k_proj_group = self.k_proj_cache[consumer_start:consumer_end, :, :, :limit, :]
        
        # Compute scores
        with self._record('compute_scores'):
            if self.quantize_int8 and HAS_INT8_FUSED_KERNEL:
                q_for_fused = q_group
                scale_group = self.k_proj_scale[consumer_start:consumer_end]
                scores = score_int8_fused(q_for_fused, k_proj_group, scale_group)
                scores = scores / math.sqrt(self.dDash)
                scores = scores.max(dim=3).values
            else:
                if self.quantize_int8:
                    scale_group = self.k_proj_scale[consumer_start:consumer_end].to(self.dtype)
                    k_proj_group = k_proj_group.to(self.dtype) * scale_group
                k_proj_f = k_proj_group.permute(1, 0, 2, 3, 4)
                scores = torch.einsum("blhgd,blhkd->blhgk", q_group, k_proj_f)
                scores = scores / math.sqrt(self.dDash)
                scores = scores.max(dim=3).values
            
            limit = scores.size(-1)
            if self.min_sparse_index > 0:
                sink = min(self.min_sparse_index, limit)
                scores[:, :, :, :sink] = float("-inf")
            
            local_start = max(0, kv_len - self.local_window)
            if local_start < limit:
                scores[:, :, :, local_start:] = float("-inf")
            
            scores = torch.softmax(scores.float(), dim=-1).to(self.dtype)
        
        # Select indices and gather from CPU
        selection_end = min(local_start, self.last_projected_pos)
        selection_start = self.min_sparse_index
        num_available = max(0, selection_end - selection_start)
        num_to_select = min(self.sparse_budget, num_available)
        
        if num_to_select > 0:
            with self._record('topk_selection'):
                _, topk_indices = torch.topk(scores, k=num_to_select, dim=-1)
                topk_indices, _ = topk_indices.sort(dim=-1)
            
            # Gather from CPU cache to GPU buffer
            sparse_start = self.min_sparse_index + self.local_window
            
            with self._record('cpu_to_gpu_gather'):
                # Wait for any pending CPU copies
                self.copy_stream.synchronize()
                
                indices_perm = topk_indices.permute(1, 0, 2, 3)  # [Lgrp, B, H, budget]
                indices_expanded = indices_perm.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
                
                # Gather from CPU and copy to GPU (layer by layer for memory efficiency)
                for i, layer_idx in enumerate(range(consumer_start, consumer_end)):
                    layer_indices = indices_expanded[i]  # [B, H, budget, head_dim]
                    
                    # Gather from CPU cache
                    k_cpu = self.k_cache[layer_idx]  # [B, H, max_len, head_dim] on CPU
                    v_cpu = self.v_cache[layer_idx]
                    
                    # Move indices to CPU for gather, then result back to GPU
                    layer_indices_cpu = layer_indices.cpu()
                    k_gathered = torch.gather(k_cpu, dim=2, index=layer_indices_cpu)
                    v_gathered = torch.gather(v_cpu, dim=2, index=layer_indices_cpu)
                    
                    # Copy to GPU buffer
                    self.k_cache_buffer[layer_idx, :, :, sparse_start:sparse_start + num_to_select].copy_(
                        k_gathered.to(self.device), non_blocking=True
                    )
                    self.v_cache_buffer[layer_idx, :, :, sparse_start:sparse_start + num_to_select].copy_(
                        v_gathered.to(self.device), non_blocking=True
                    )
                
                # Sync to ensure transfers complete before attention
                torch.cuda.synchronize()
    
    def get_retrieval_position_ids(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """No-op for batched KeySifter."""
        return None
    
    def get_key_cache(
        self,
        layer_idx: int,
        position_ids: Optional[torch.Tensor] = None,
        rope_func: Callable = None,
        cos_sin_cache: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Return the key cache for attention.
        
        Layer 0 and first decode token use full cache (transferred from CPU).
        Other layers use GPU buffer with sparse selection.
        """
        kv_len = self._kv_len_eff()
        
        # Layer 0 or first decode: need full cache from CPU
        if layer_idx == 0 or kv_len <= self._dense_decode_cutoff:
            # Transfer full cache from CPU to GPU
            return self.k_cache[layer_idx, :, :, :kv_len].to(self.device)
        
        # Calculate buffer length
        local_start = max(0, kv_len - self.local_window)
        selection_end = min(local_start, self.last_projected_pos)
        selection_start = self.min_sparse_index
        num_available = max(0, selection_end - selection_start)
        num_selected = min(self.sparse_budget, num_available)
        
        total_len = self.min_sparse_index + self.local_window + num_selected
        
        return self.k_cache_buffer[layer_idx, :, :, :total_len]
    
    def get_value_cache(
        self,
        layer_idx: int,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return the value cache for attention.
        
        Layer 0 and first decode token use full cache (transferred from CPU).
        Other layers use GPU buffer with sparse selection.
        """
        kv_len = self._kv_len_eff()
        
        # Layer 0 or first decode: need full cache from CPU
        if layer_idx == 0 or kv_len <= self._dense_decode_cutoff:
            return self.v_cache[layer_idx, :, :, :kv_len].to(self.device)
        
        # Calculate buffer length
        local_start = max(0, kv_len - self.local_window)
        selection_end = min(local_start, self.last_projected_pos)
        selection_start = self.min_sparse_index
        num_available = max(0, selection_end - selection_start)
        num_selected = min(self.sparse_budget, num_available)
        
        total_len = self.min_sparse_index + self.local_window + num_selected
        
        return self.v_cache_buffer[layer_idx, :, :, :total_len]
