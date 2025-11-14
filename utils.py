import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import gc
import matplotlib.pyplot as plt

import traceback
import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from scipy.stats import spearmanr
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, LlamaRMSNorm, apply_rotary_pos_emb
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import matplotlib.cm as cm
from scipy.spatial.distance import cosine
from tqdm import tqdm

class SlidingWindowCache:
    def __init__(self, max_seq_len, sliding_window, device):
        self.sliding_window = sliding_window
        self.device = device
        if sliding_window is None:
            self.max_seq_len = 0
            self.window = None
        else:
            self.max_seq_len = max_seq_len
            self.window = self._create_window(self.max_seq_len)

    def _create_window(self, seq_len):
        idx = torch.arange(seq_len, device=self.device)
        query = idx.unsqueeze(1)  # [seq_len, 1]
        key = idx.unsqueeze(0)    # [1, seq_len]
        win = (key >= (query - self.sliding_window + 1)) & (key <= query)
        return win.unsqueeze(0).unsqueeze(0)  # [1,1,seq_len,seq_len]

    def get_window(self, q_len, key_len):
        if self.sliding_window is None:
            return None
        req = max(q_len, key_len)
        # Always guarantee the cache is large enough
        if req > self.max_seq_len:
            self.max_seq_len = req
            self.window = self._create_window(self.max_seq_len)

        # Select the correct rows: the queries are the *last* q_len tokens
        row_start = key_len - q_len
        row_end   = key_len
        return self.window[:, :, row_start:row_end, :key_len]

def enforce_sliding_window(mask_tensor, window):
    if window is None:
        return mask_tensor
    return mask_tensor.masked_fill(window, 0.0)


def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*\'\[\]]', '_', name)

def args_to_name(args, timestamp=True):
    args_dict = vars(args)
    args_dict = args_dict.copy()
    # remove longbench_datasets, task_list from args_dict
    args_dict.pop("longbench_datasets", None)
    args_dict.pop("task_list", None)
    model_descr = list(args_dict.values())
    # Split the model description into two parts
    split_point = len(model_descr) // 2
    folder_part = model_descr[:split_point]
    file_part = model_descr[split_point:]
    # Create a sanitized folder name from the first part
    folder_name = "_".join([str(elem) for elem in folder_part])
    folder_name = sanitize_filename(folder_name)
    # Create a sanitized file name from the second part
    file_name = "_".join([str(elem) for elem in file_part])
    file_name = sanitize_filename(file_name)
    # Add timestamp to file name
    if timestamp:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_name = file_name + "_" + timestamp
    return folder_name, file_name

    
def snapkv_mask_only(self, query_states, key_states, value_states, attention_mask=None):
    """
    'Mask-only' version of SnapKV that does not gather/slice the actual key_states:
    - If q_len < max_capacity_prompt, do nothing.
    - Else, we compute the 'top prefix tokens' using the last window_size queries, 
      plus the last window_size tokens themselves.
    - Then we create a single-step mask that is -inf for all other tokens.

    We store that single-step mask in self.snapkv_cache so that 
    on the next decode step (q_len=1) we can re-apply it.
    """
    bsz, num_heads, q_len, head_dim = query_states.shape
    # Ensure prefix-phase
    assert key_states.shape[-2] == query_states.shape[-2], "Prefix shape mismatch"

    # If no compression is needed, just return the normal outputs
    if q_len < self.max_capacity_prompt:
        return None  # signals: no special mask built

    # 1) Compute local attention (like SnapKV: last window_size queries vs entire prefix)
    obs = self.window_size
    if obs > q_len:
        obs = q_len  # if the prompt is shorter than window_size
    attn_logits = torch.matmul(query_states[..., -obs:, :],
                               key_states.transpose(-2, -1)) / math.sqrt(head_dim)
    # shape [bsz, num_heads, obs, kv_seq_len]

    # 2) Build a local triangular mask of shape (obs, obs) for the last window_size queries
    mask = torch.full((obs, obs), float('-inf'), device=attn_logits.device, dtype=attn_logits.dtype)
    idxs = torch.arange(obs, device=mask.device)
    mask.masked_fill_(idxs < idxs.unsqueeze(-1), 0)  # lower-tri (including diagonal)=0, above diag=-inf
    local_mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,obs,obs]

    # Apply it to the last obs block in attn_logits
    attn_logits[:, :, -obs:, -obs:] += local_mask

    # 3) Softmax
    attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # shape [bsz, num_heads, obs, kv_seq_len]

    # 4) Sum across the obs dimension => [bsz, num_heads, kv_seq_len]
    attn_sum = attn_probs.sum(dim=-2)

    # 5) Optional pooling => must pass [N, C, L] to max_pool1d/avg_pool1d
    # attn_sum is shape [bsz, num_heads, kv_seq_len]. We can flatten bsz*num_heads
    bnh = bsz * num_heads
    L = key_states.shape[-2]  # kv_seq_len
    x = attn_sum.view(bnh, 1, L)  # [bnh, 1, kv_seq_len]

    if self.pooling == 'avgpool':
        pooled = F.avg_pool1d(
            x,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2
        )
    elif self.pooling == 'maxpool':
        pooled = F.max_pool1d(
            x,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2
        )
    else:
        raise ValueError("Unsupported pooling method")

    # Now pooled is shape [bnh, 1, L']
    # Usually, L' = L if stride=1, but let's just keep it so it lines up with kv_seq_len
    pooled = pooled.view(bsz, num_heads, -1)  # [bsz, num_heads, kv_seq_len]

    # 6) topk
    top_prefix_to_keep = self.max_capacity_prompt - obs
    prefix_indices = pooled.topk(top_prefix_to_keep, dim=-1).indices  # [bsz, num_heads, top_prefix_to_keep]

    # 7) Build single-step mask => shape [bsz, num_heads, 1, kv_seq_len]
    single_mask = torch.full(
        (bsz, num_heads, 1, L),
        float('-inf'),
        device=query_states.device,
        dtype=query_states.dtype
    )

    # unmask the top prefix positions
    row_idx = torch.arange(bsz, device=query_states.device).view(bsz, 1, 1)
    head_idx = torch.arange(num_heads, device=query_states.device).view(1, num_heads, 1)
    row_idx = row_idx.expand(bsz, num_heads, top_prefix_to_keep)
    head_idx = head_idx.expand(bsz, num_heads, top_prefix_to_keep)
    single_mask[row_idx, head_idx, 0, prefix_indices] = 0.0

    # unmask the last obs tokens
    single_mask[:, :, 0, -obs:] = 0.0

    return single_mask

def calculate_effective_sparsity(final_mask, attention_mask):
    true_mask = final_mask + attention_mask
    num_deact = true_mask.bool().sum(dim=-1)                   # Number of tokens disabled.
    causally_deact = (attention_mask.bool()).sum(dim=-1).expand_as(num_deact)        # Number of tokens disabled causally anyway
    additional_deact = (num_deact - causally_deact)
    num_active = (~attention_mask.bool()).sum(dim=-1).expand_as(num_deact)    # Number of tokens active at this position if zero-sparsity
    effective_sparsity = 100 * (additional_deact.float() / num_active.float()).mean().item()
    return effective_sparsity


def sorted_index_to_variable_mask(
    sorted_indices,          # [B,H,Lq,Lk]
    attention_mask,          # [1,1,Lq,Lk]
    min_sparse_index,
    bsz, q_len, key_len,
    head_keep,               # 1‑D tensor [H] with keep fractions
    sliding_window=None
):
    device = sorted_indices.device
    H = sorted_indices.size(1)           # num heads (32 in your trace)

    # (1) put head_keep into a broadcast‑ready shape
    head_keep = head_keep.view(1, H, 1, 1)          # [1, H, 1, 1]

    # (2) query positions (same as before, just cleaner)
    if q_len == 1:
        query_pos = torch.full((1, 1, 1, 1), key_len + 1, device=device)
    else:
        query_pos = torch.arange(1, q_len + 1, device=device).view(1, 1, q_len, 1)

    # (3) per‑head budget K   →  shape [1, H, Lq, 1]
    K = torch.ceil(query_pos * head_keep).long().clamp(max=key_len)
    K = torch.clamp(K, max=key_len)

    # Step 2: Expand attention_mask to [B,H,q_len,key_len]
    attention_mask_expanded = attention_mask.expand(bsz, -1, -1, -1)
    attention_mask_expanded = attention_mask_expanded.expand(-1, sorted_indices.size(1), -1, -1)
    # Convert True -> 1, False -> 0
    attention_mask_expanded = (~attention_mask_expanded.bool()).int()

    # Step 3: Gather (reorder) mask by sorted_indices
    gathered_mask = torch.gather(attention_mask_expanded, dim=-1, index=sorted_indices)

    # Step 4: cumsum along sorted dimension
    gathered_mask_float = gathered_mask.float()
    cum_sum = torch.cumsum(gathered_mask_float, dim=-1)  # [B,H,q_len,key_len]

    # Step 5: Compare cumsum <= K
    # Expand K to [B,H,q_len,key_len] for broadcast
    
    # K_broadcast = K.view(1, 1, q_len, 1).expand_as(cum_sum)
    # selected_mask = (cum_sum <= K_broadcast)
    K_original = torch.ceil(query_pos * head_keep).long()
    K_adjusted = torch.clamp(K_original - min_sparse_index, min=0, max=key_len)
    selected_mask = cum_sum <= K_adjusted

    # Step 6: Prepare final mask_tensor with -inf by default
    mask_tensor = torch.full_like(attention_mask_expanded.float(), float('-inf'))

    # Step 7: Scatter 0 where selected, -inf otherwise
    scatter_values = torch.zeros_like(gathered_mask_float)
    scatter_values = scatter_values.masked_fill(~selected_mask, float('-inf'))
    mask_tensor.scatter_(-1, sorted_indices, scatter_values)

    # Step 8: Force the guaranteed front region unmasked
    mask_tensor[:, :, :, :min_sparse_index] = 0.0

    # We do NOT forcibly unmask the trailing `sliding_window` here,
    # because we typically do it with a separate function that
    # ensures the last `sliding_window` positions are unmasked for each query.
    # Replace with self.sliding_window where referenced
    # Where not referenced, reduce budget in calculation.

    return mask_tensor


def sorted_index_to_mask(
    sorted_indices,
    attention_mask,
    min_sparse_index,
    bsz,
    q_len,
    key_len,
    sparse_aggression,
    sliding_window=None
):
    """
    sorted_indices: [B, H, q_len, key_len]
    attention_mask: [1, 1, q_len, key_len]  (True = keep, False = mask out, or vice versa)
    min_sparse_index: guaranteed front region to keep
    sliding_window: guaranteed trailing region (for each query) to keep
    sparse_aggression: float in [0,1], fraction of keys to drop or keep
    """
    device = sorted_indices.device
    dtype = sorted_indices.dtype

    # Step 1: Compute base K
    if q_len == 1:  
        query_positions = torch.arange(q_len, device=device).view(1, 1, q_len, 1).float()
        query_positions[0] = key_len + 1
    else:
        query_positions = torch.arange(q_len, device=device).view(1, 1, q_len, 1).float() + 1.0
    K_original = torch.ceil(query_positions * sparse_aggression).long()  # [1,1,q_len,1]
    K_original = torch.clamp(K_original, max=key_len)
    # Step 1b: subtract *only* the guaranteed prefix.
    # The trailing sliding‑window region is enforced later via a
    # separate mask and therefore must **not** reduce K here.
    guaranteed_prefix = min_sparse_index
    K_adjusted = torch.clamp(
        K_original - guaranteed_prefix,
        min=0,
        max=key_len
    )

    # Step 2: Expand attention_mask to [B,H,q_len,key_len]
    attention_mask_expanded = attention_mask.expand(bsz, -1, -1, -1)
    attention_mask_expanded = attention_mask_expanded.expand(-1, sorted_indices.size(1), -1, -1)
    # Convert True -> 1, False -> 0
    attention_mask_expanded = (~attention_mask_expanded.bool()).int()

    # Step 3: Gather (reorder) mask by sorted_indices
    gathered_mask = torch.gather(attention_mask_expanded, dim=-1, index=sorted_indices)

    # Step 4: cumsum along sorted dimension
    gathered_mask_float = gathered_mask.float()
    cum_sum = torch.cumsum(gathered_mask_float, dim=-1)  # [B,H,q_len,key_len]

    # Step 5: Compare cumsum <= K_adjusted
    # Expand K_adjusted to [B,H,q_len,key_len] for broadcast
    K_broadcast = K_adjusted.view(1, 1, q_len, 1).expand_as(cum_sum)
    selected_mask = (cum_sum <= K_broadcast)

    # Step 6: Prepare final mask_tensor with -inf by default
    mask_tensor = torch.full_like(attention_mask_expanded.float(), float('-inf'))

    # Step 7: Scatter 0 where selected, -inf otherwise
    scatter_values = torch.zeros_like(gathered_mask_float)
    scatter_values = scatter_values.masked_fill(~selected_mask, float('-inf'))
    mask_tensor.scatter_(-1, sorted_indices, scatter_values)

    # Step 8: Force the guaranteed front region unmasked
    mask_tensor[:, :, :, :min_sparse_index] = 0.0

    # We do NOT forcibly unmask the trailing `sliding_window` here,
    # because we typically do it with a separate function that
    # ensures the last `sliding_window` positions are unmasked for each query.
    # Replace with self.sliding_window where referenced
    # Where not referenced, reduce budget in calculation.

    return mask_tensor

def threshold_to_mask(unadj_importance_mask, perhead_thresholds, min_sparse_index, bsz, q_len, key_len):
    """
    Create a mask tensor based on per-head thresholds, setting values below the threshold to -inf.
    
    Args:
    - unadj_importance_mask: torch.Tensor of shape [B, H, Lq, Lk].
    - perhead_thresholds: torch.Tensor of shape [H], per-head thresholds.
    - min_sparse_index: Minimum index for sparsity; values below this index will not be masked.
    - bsz: Batch size.
    - q_len: Query length (Lq).
    - key_len: Key length (Lk).

    Returns:
    - mask_tensor: torch.Tensor of shape [B, H, Lq, Lk], with values below threshold as -inf.
    """
    # Ensure perhead_thresholds is in the correct shape for broadcasting
    thresholds_broadcast = perhead_thresholds.view(1, -1, 1, 1)  # [1, H, 1, 1]

    # Compare unadj_importance_mask with thresholds to create a mask
    mask_tensor = torch.where(
        unadj_importance_mask >= thresholds_broadcast, 
        torch.zeros_like(unadj_importance_mask), 
        torch.full_like(unadj_importance_mask, float('-inf'))
    )  # [B, H, Lq, Lk]

    # Ensure mask_tensor has mask_tensor[:, :, :, :min_sparse_index] = 0
    mask_tensor[:, :, :, :min_sparse_index] = 0.0

    return mask_tensor

def calculate_hit_metrics(estimated_importance: torch.Tensor, 
                          true_importance: torch.Tensor, 
                          top_k_ratio: float = 0.5) -> Tuple[float, float, float]:
    """
    Calculate hit accuracy, mean, and max rank correlation between estimated and true importance tensors.
    We compute metrics along the last dimension of the input tensors.

    Shapes:
      - 4D token-importance: [B, H, L, L]. We slice the last query (index -1) => [B, H, L].
      - 3D head-importance:  [B, L, H]. We use all of it as-is => [B, L, H].
    
    Args:
        estimated_importance (torch.Tensor): [B, H, L, L] or [B, L, H]
        true_importance      (torch.Tensor): [B, H, L, L] or [B, L, H]
        top_k_ratio (float): Fraction of top-k elements to consider for hit accuracy (default=0.5).
    
    Returns:
        (hit_accuracy, mean_corr, max_corr):
            hit_accuracy (float): Intersection ratio of top-k sets (0..1).
            mean_corr (float): Average Spearman rank correlation over all [B, ...].
            max_corr (float): Maximum Spearman rank correlation among all [B, ...].
    """

    # 1) Standardize shapes so the last dimension is what we rank over.
    if estimated_importance.dim() == 4:
        # Shape is [B, H, L, L] => slice to keep only the last query => [B, H, L]
        estimated_importance = estimated_importance[:, :, -1, :]
        true_importance      = true_importance[:, :, -1, :]
        # after slicing: [B, H, L]
        # For intersection denominator => top_k * B * H
        denom_for_hits = estimated_importance.size(0) * estimated_importance.size(1)
    elif estimated_importance.dim() == 3:
        # Shape is [B, L, H], the last dimension is H
        # For intersection denominator => top_k * B * L
        denom_for_hits = estimated_importance.size(0) * estimated_importance.size(1)
    else:
        raise ValueError("Tensors must be either 4D [B,H,L,L] or 3D [B,L,H].")

    # 2) Compute Spearman rank correlation along the last dimension.
    #    Sort indices in descending order => get 'ranks' for correlation.
    _, sorted_esti = torch.sort(estimated_importance, dim=-1, descending=True)
    _, sorted_true = torch.sort(true_importance, dim=-1, descending=True)

    # Spearman's rho = 1 - 6 * sum(d^2) / [n*(n^2 - 1)]
    n = sorted_esti.shape[-1]
    d = sorted_esti.float() - sorted_true.float()
    d_squared = d ** 2
    sum_d_squared = d_squared.sum(dim=-1)
    rank_corr = 1 - (6 * sum_d_squared) / (n * (n**2 - 1))  # shape: [B,H] or [B,L]

    mean_corr = rank_corr.mean().item()
    max_corr  = rank_corr.max().item()

    # 3) Compute top-k hit accuracy along the last dimension.
    top_k = max(1, int(n * top_k_ratio))
    _, top_esti_indices = torch.topk(estimated_importance, top_k, dim=-1)
    _, top_true_indices = torch.topk(true_importance,      top_k, dim=-1)

    # top_esti_indices => [B,H,top_k] or [B,L,top_k]
    # top_true_indices => [B,H,top_k] or [B,L,top_k]
    # matches => [B,H,top_k,top_k] or [B,L,top_k,top_k]
    matches = (top_esti_indices.unsqueeze(-1) == top_true_indices.unsqueeze(-2))
    intersection = matches.any(dim=-1).sum(dim=-1)  # => [B,H] or [B,L]

    # Each [B,H] or [B,L] element can have at most 'top_k' matches, so total is top_k * denom_for_hits.
    total_possible = top_k * denom_for_hits
    hit_accuracy = intersection.sum().item() / total_possible  # => 0..1

    return hit_accuracy, mean_corr, max_corr

def plot_thresholds(threshold_tensor, true_threshold_tensor, fpath_base, fpath_specific):
    """
    Plots mean and error regions for random layers and heads, showing threshold changes across decode steps.
    
    Args:
    - threshold_tensor: torch.Tensor of shape [163, 31, 32, 1024].
    - true_threshold_tensor: torch.Tensor of shape [163, 31, 32, 1024].
    """
    def create_plot(tensor, title, filename):
        """
        Helper function to generate the plot.
        """
        # Choose 3 random layers
        layers = np.random.choice(tensor.shape[1], 3, replace=False)
        # layers = np.array([0, 15, 30])
        
        # Create subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        x = np.arange(tensor.shape[3])  # Decode steps (1024)
        
        for i, layer in enumerate(layers):
            # Choose 5 random heads for this layer
            heads = np.random.choice(tensor.shape[2], 5, replace=False)
            
            for head in heads:
                try:
                    # Extract data for the selected layer and head
                    data = tensor[:, layer, head, :].numpy()  # Shape [163, 1024]
                    
                    # Compute mean and standard deviation across samples (dim=0)
                    mean = np.mean(data, axis=0)
                    std = np.std(data, axis=0)
                    
                    # Plot mean and shaded error region for the head
                    axs[i].plot(x, mean, label=f"Head {head}")
                    axs[i].fill_between(x, mean - std, mean + std, alpha=0.3)
                except:
                    import pdb; pdb.set_trace()
            
            # Customize subplot
            axs[i].set_title(f"Layer {layer}")
            axs[i].set_xlabel("Decode Step")
            axs[i].grid(True)
            axs[i].legend(fontsize=8)  # Adjust legend size for multiple heads
        
        # Common Y-axis label and adjustments
        axs[0].set_ylabel("Threshold")
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot
        plt.savefig(filename)
        plt.close()

    def compute_mean_threshold(tensor):
        """
        Computes the mean threshold value for each head and layer, excluding the first 32 tokens.
        """
        # Exclude the first 32 tokens (dimension 1024)
        tensor_excluded = tensor[:, :, :, 32:]  # Shape [163, 31, 32, 992]
        
        # Compute the mean along the first (samples) and last (remaining tokens) dimensions
        mean_threshold = tensor_excluded.mean(dim=(0, -1))  # Shape [31, 32]
        
        return mean_threshold

    # create folder fpath_base if it does not exist
    if not os.path.exists(f"threshold_plots"):
        os.makedirs(f"threshold_plots")
    if not os.path.exists(f"threshold_plots/{fpath_base}"):
        os.makedirs(f"threshold_plots/{fpath_base}")
    # Plot for threshold_tensor
    create_plot(threshold_tensor, "Post-Attention Thresholds", f"threshold_plots/{fpath_base}/{fpath_specific}_postattn_threshold.pdf")
    
    # Plot for true_threshold_tensor
    create_plot(true_threshold_tensor, "Predicted Pre-SM Thresholds", f"threshold_plots/{fpath_base}/{fpath_specific}_pred_presm_threshold.pdf")

    # Compute mean thresholds
    mean_threshold_postattn = compute_mean_threshold(threshold_tensor)
    mean_threshold_predpresm = compute_mean_threshold(true_threshold_tensor)
    
    return mean_threshold_postattn, mean_threshold_predpresm


    
class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0, config=None):
        self.scaling_factor = scaling_factor
        super().__init__(config)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0, config=None):
        self.scaling_factor = scaling_factor
        super().__init__(config)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class FlattenedDataset(Dataset):
    def __init__(self, dataset, max_seq_len, max_repeat_fraction=0.7):
        self.max_seq_len = max_seq_len
        self.max_repeat_fraction = max_repeat_fraction

        # Extract and flatten the input_ids column
        all_tokens = torch.cat([torch.tensor(ids) for ids in dataset["input_ids"]], dim=0)

        # Calculate the number of full chunks
        num_full_chunks = len(all_tokens) // self.max_seq_len
        all_chunks = all_tokens[:num_full_chunks * self.max_seq_len].view(-1, self.max_seq_len)

        # Filter out chunks with excessive repeated tokens
        self.chunks = []
        for chunk in all_chunks:
            unique_tokens, counts = torch.unique(chunk, return_counts=True)
            max_repeats = counts.max().item()
            if max_repeats <= self.max_repeat_fraction * chunk.numel():
                self.chunks.append(chunk)
        
        self.chunks = torch.stack(self.chunks)  # Stack the remaining chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def compute_js_divergence(p, q, epsilon=1e-12):
    """
    Compute the Jensen-Shannon Divergence between two probability distributions.
    
    Args:
        p (torch.Tensor): Shape [..., D]
        q (torch.Tensor): Shape [..., D]
        epsilon (float): Small value to avoid log(0)
    
    Returns:
        torch.Tensor: JS Divergence values per pair (Shape: [...])
    """
    # Add epsilon to avoid log(0)
    p = p + epsilon
    q = q + epsilon
    
    # Normalize to ensure they are valid probability distributions
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    
    # Compute the average distribution
    m = 0.5 * (p + q)
    
    # Compute KL Divergences
    kl_p = F.kl_div(m.log(), p, reduction='none').sum(dim=-1)
    kl_q = F.kl_div(m.log(), q, reduction='none').sum(dim=-1)
    
    # Compute JS Divergence
    js = 0.5 * kl_p + 0.5 * kl_q
    
    return js

def compute_head_consistency_js(head_data):
    """
    Compute the consistency of a head's probability distributions across examples using JS Divergence.
    
    Args:
        head_data (torch.Tensor): Shape [163, 1024], probability distributions for one head.
    
    Returns:
        float: Mean pairwise JS Divergence across examples.
    """
    num_examples = head_data.size(0)
    
    # Ensure head_data is float for division and normalization
    head_data = head_data.float()
    
    # Normalize each distribution to ensure it sums to 1
    head_data = head_data / head_data.sum(dim=-1, keepdim=True)
    
    # Initialize variables to accumulate JS divergences
    total_js = 0.0
    count = 0
    
    # Iterate over all unique pairs without redundancy
    for i in tqdm(range(num_examples), desc="Computing JS Divergence"):
        # Select the i-th distribution and expand its dimensions
        p = head_data[i].unsqueeze(0)  # Shape: [1, D]
        
        # Select all distributions after the i-th to avoid duplicate pairs
        q = head_data[i+1:]  # Shape: [N-i-1, D]
        
        if q.size(0) == 0:
            continue  # No more pairs left
        
        # Compute JS Divergence between p and q
        js = compute_js_divergence(p.repeat(q.size(0), 1), q)  # Shape: [N-i-1]
        
        # Accumulate the sum of JS Divergence
        total_js += js.sum().item()
        count += js.size(0)
    
    # Compute the mean JS Divergence
    mean_js = total_js / count if count > 0 else 0.0
    
    return mean_js
    
    
def compute_token_consistency_js(head_data):
    """
    Compute token consistency for all heads in a layer using JS Divergence.
    
    Args:
        head_data (torch.Tensor): Shape [163, 24, 1024], layer's head data.

    Returns:
        np.ndarray: Consistency values for all 24 heads.
    """
    num_heads = head_data.shape[1]
    consistency_metrics = []
    
    for head in tqdm(range(num_heads), desc="Processing Heads"):
        head_consistency = compute_head_consistency_js(head_data[:, head, :])  # Consistency for one head
        consistency_metrics.append(head_consistency)
    
    return np.array(consistency_metrics)


def graph_headtok_pos_affinity(head_tokpos_affinity, args):
    """
    Generate a violin plot for Token Access Consistency Across Layers using JS Divergence.
    
    Args:
        head_tokpos_affinity (dict): Dictionary where keys are layer identifiers and values are 
                                     torch.Tensor of shape [163, 24, 1024].
        args (argparse.Namespace): Arguments containing at least 'model_path'.
    """
    # Process the data into a format suitable for plotting
    layer_ids = []
    consistency_values = []
    
    for layer, tensor in tqdm(head_tokpos_affinity.items(), desc="Processing Layers"):
        layer_consistency = compute_token_consistency_js(tensor)  # Shape: [24]
        layer_ids.extend([layer] * len(layer_consistency))
        consistency_values.extend(layer_consistency)

    # Create directory structure: ablation_plots/traces/tok_js_div
    trace_dir = f"ablation_plots/traces/tok_js_div"
    os.makedirs(trace_dir, exist_ok=True)
    mpath = args.model_path.replace("/", "_")
    # Save data to a .npy file (NumPy array format)
    trace_path = os.path.join(trace_dir, f"layer_consistency_{mpath}.npy")
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    np.save(trace_path, {"Layer": layer_ids, "JS_Divergence": consistency_values})
    print(f"Consistency data saved to {trace_path}")

    
    # Prepare data for Seaborn violin plot
    data = {"Layer": layer_ids, "JS_Divergence": consistency_values}
    
    # Create the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=data["Layer"], y=data["JS_Divergence"], scale="width", inner="quartile", palette="viridis")
    
    # Formatting the plot
    plt.title(f"Token Access Consistency Across Layers for {args.model_path}", fontsize=16)
    plt.xlabel("Layer", fontsize=14)
    plt.ylabel("Token Consistency Metric (Mean JS Divergence)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Enhance layout
    plt.tight_layout()
    
    # Create ablation_plots directory if it doesn't exist
    os.makedirs("ablation_plots", exist_ok=True)
    
    # Construct the full file path
    file_path = f"ablation_plots/{mpath}_headtok_consistency_js_divergence.pdf"
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Save the plot
    plt.savefig(file_path)
    plt.close()


def compute_head_agreement_js(head_data):
    """
    Compute head agreement for a single example using JS Divergence.
    
    Args:
        head_data (torch.Tensor): Shape [num_heads, num_tokens], token distributions for all heads.
    
    Returns:
        float: Mean pairwise JS Divergence for the heads.
    """
    num_heads = head_data.size(0)
    js_divergences = []

    for i in range(num_heads):
        p = head_data[i].unsqueeze(0)  # Shape: [1, num_tokens]
        q = head_data[i + 1 :]  # Remaining heads, Shape: [num_heads - i - 1, num_tokens]
        
        if q.size(0) == 0:
            continue
        
        js = compute_js_divergence(p.repeat(q.size(0), 1), q)  # Pairwise JS Div
        js_divergences.extend(js.tolist())

    # Return the mean of the upper triangular JS Divergence matrix
    return np.mean(js_divergences)

def compute_head_agreement_all_examples(head_tokpos_affinity):
    """
    Compute head agreement for all examples across all layers.
    
    Args:
        head_tokpos_affinity (dict): Dictionary where keys are layer identifiers and values are 
                                     torch.Tensor of shape [num_examples, num_heads, num_tokens].

    Returns:
        np.ndarray: Head agreement values for all examples.
    """
    agreement_values = []

    for layer, tensor in tqdm(head_tokpos_affinity.items(), desc="Processing Layers"):
        for example_idx in range(tensor.shape[0]):  # Iterate over examples
            head_data = tensor[example_idx]  # Shape: [num_heads, num_tokens]
            agreement = compute_head_agreement_js(head_data)
            agreement_values.append(agreement)

    return np.array(agreement_values)

def plot_and_save_head_agreement(agreement_values, args):
    """
    Save head agreement values and plot them as a violin plot.
    
    Args:
        agreement_values (np.ndarray): Head agreement values for all examples.
        args (argparse.Namespace): Arguments containing at least 'model_path'.
    """
    # Save agreement values to a .npy file
    trace_dir = "ablation_plots/traces/headagreement_js_div"
    os.makedirs(trace_dir, exist_ok=True)
    mpath = args.model_path.replace("/", "_")
    trace_path = os.path.join(trace_dir, f"head_agreement_{mpath}.npy")
    np.save(trace_path, {"HeadAgreement": agreement_values})
    print(f"Head agreement values saved to {trace_path}")

    # Plot the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=[agreement_values], scale="width", inner="quartile", palette="viridis")

    # Formatting
    plt.title(f"Head Agreement Across Examples for {mpath}", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Mean JS Divergence Between Heads", fontsize=14)
    plt.xticks([0], [mpath], fontsize=12)  # Single violin
    plt.yticks(fontsize=12)

    # Enhance layout
    plt.tight_layout()

    # Save the plot
    plot_path = f"ablation_plots/{mpath}_head_agreement_js_divergence.pdf"
    plt.savefig(plot_path)
    print(f"Violin plot saved to {plot_path}")
    plt.close()


def compute_jsd_over_decode_steps(decode_probs):
    """
    Compute average JSD over decode steps for a single head.
    
    Args:
        decode_probs (torch.Tensor): Shape [50, 974], softmaxed token importances for 50 decode steps.
    
    Returns:
        float: Mean JSD across the upper diagonal of the pairwise JSD matrix.
    """
    # Expand dims to create pairwise matrices
    p = decode_probs.unsqueeze(0)  # Shape: [1, 50, 974]
    q = decode_probs.unsqueeze(1)  # Shape: [50, 1, 974]
    
    # Compute pairwise JSD (broadcasting handles the pairwise combinations)
    jsd_matrix = compute_js_divergence(p, q)  # Shape: [50, 50]
    
    # Extract upper diagonal without the diagonal itself
    triu_indices = torch.triu_indices(jsd_matrix.size(0), jsd_matrix.size(1), offset=1)
    jsd_upper = jsd_matrix[triu_indices[0], triu_indices[1]]  # Shape: [N*(N-1)/2]
    
    return jsd_upper.mean().item()

def compute_layer_jsd(decode_tokpos_affinity):
    """
    Compute average JSD over decode steps for all heads and layers.
    
    Args:
        decode_tokpos_affinity (dict): Dictionary where keys are layer indices and values are 
                                       torch.Tensor of shape [163, 24, 50, 974].
    
    Returns:
        dict: Average JSD values for each head in each layer, keyed by layer index.
    """
    layer_jsd = {}

    for layer, tensor in tqdm(decode_tokpos_affinity.items(), desc="Processing Layers"):
        # tensor shape: [163, 24, 50, 974]
        num_examples, num_heads, num_decode_steps, num_tokens = tensor.shape
        
        # Reshape for batch processing
        decode_probs = tensor.view(-1, num_decode_steps, num_tokens)  # Shape: [163 * 24, 50, 974]
        
        # Compute pairwise JSD for all heads and examples
        jsd_values = []
        for decode_head in decode_probs:
            jsd_values.append(compute_jsd_over_decode_steps(decode_head))

        # Reshape back to per-head per-layer values
        jsd_values = torch.tensor(jsd_values).view(num_examples, num_heads)  # Shape: [163, 24]
        layer_jsd[layer] = jsd_values.mean(dim=0).tolist()  # Average across examples per head

    return layer_jsd

def plot_decode_jsd_violin(layer_jsd, args):
    """
    Plot per-layer JSD values as violins.
    
    Args:
        layer_jsd (dict): Dictionary of average JSD values for each head in each layer.
        args (argparse.Namespace): Arguments containing at least 'model_path'.
    """
    # Prepare data for per-layer violin plot
    layers = []
    jsd_values = []

    for layer, values in layer_jsd.items():
        layers.extend([layer] * len(values))
        jsd_values.extend(values)

    # Save JSD values to a file
    trace_dir = "ablation_plots/traces/decode_jsd"
    os.makedirs(trace_dir, exist_ok=True)
    mpath = args.model_path.replace("/", "_")
    trace_path = os.path.join(trace_dir, f"decode_jsd_{mpath}.npy")
    np.save(trace_path, {"Layer": layers, "JSD": jsd_values})
    print(f"Decode JSD data saved to {trace_path}")

    # Plot the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=layers, y=jsd_values, scale="width", inner="quartile", palette="viridis")

    # Formatting
    plt.title(f"Per-Layer Decode JSD for {args.model_path}", fontsize=16)
    plt.xlabel("Layer", fontsize=14)
    plt.ylabel("Average JSD Over Decode Steps", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()

    # Save the plot
    plot_path = f"ablation_plots/{mpath}_decode_jsd_per_layer.pdf"
    plt.savefig(plot_path)
    print(f"Violin plot saved to {plot_path}")
    plt.close()

def compute_percentage_match_vectorized(decode_probs, top_k=0.1):
    """
    Compute the average percentage match of top-k token indices across 50 decode steps for a single head.
    
    Args:
        decode_probs (torch.Tensor): Shape [50, 974], softmaxed token importances for 50 decode steps.
        top_k (float): Percentage of top tokens to consider (e.g., 0.1 for top 10%).
    
    Returns:
        float: Average percentage match of token indices across the 50 decode steps.
    """
    num_steps, num_tokens = decode_probs.shape
    k = int(num_tokens * top_k)  # Number of top tokens to consider

    # Get top-k indices for all steps
    top_indices = torch.topk(decode_probs, k, dim=-1).indices  # Shape: [50, k]

    # Create a binary mask for top-k tokens
    binary_mask = torch.zeros(num_steps, num_tokens, device=decode_probs.device)
    binary_mask.scatter_(1, top_indices, 1)  # Shape: [50, 974]

    # Compute overlap between all pairs of steps
    overlap_matrix = torch.matmul(binary_mask, binary_mask.T)  # Shape: [50, 50]
    overlap_per_pair = overlap_matrix / k  # Normalize by k to get match percentages

    # Extract upper triangular without diagonal
    triu_indices = torch.triu_indices(num_steps, num_steps, offset=1)
    upper_triangle = overlap_per_pair[triu_indices[0], triu_indices[1]]  # Shape: [N*(N-1)/2]

    return torch.tensor([upper_triangle.mean().item()], device=decode_probs.device)



def compute_layer_percentage_match_vectorized(decode_tokpos_affinity, top_k=0.1):
    """
    Compute average percentage match for top-k token indices across decode steps for all heads and layers.
    
    Args:
        decode_tokpos_affinity (dict): Dictionary where keys are layer indices and values are 
                                       torch.Tensor of shape [163, 24, 50, 974].
        top_k (float): Percentage of top tokens to consider (e.g., 0.1 for top 10%).
    
    Returns:
        dict: Average percentage match values for each head in each layer, keyed by layer index.
    """
    layer_match = {}

    for layer, tensor in tqdm(decode_tokpos_affinity.items(), desc="Processing Layers"):
        # tensor shape: [163, 24, 50, 974]
        num_examples, num_heads, num_decode_steps, num_tokens = tensor.shape

        # Flatten examples and heads for batch processing
        decode_probs = tensor.view(-1, num_decode_steps, num_tokens)  # Shape: [163 * 24, 50, 974]

        # Vectorized computation for all decode heads
        match_values = torch.cat([
            compute_percentage_match_vectorized(decode_head, top_k=top_k) for decode_head in decode_probs
        ])

        # Reshape back to per-head per-layer values
        match_values = match_values.view(num_examples, num_heads)  # Shape: [163, 24]
        layer_match[layer] = match_values.mean(dim=0).tolist()  # Average across examples per head

    return layer_match

def plot_decode_percdrift_vectorized(layer_match, args):
    """
    Plot per-layer average percentage match of top-k token indices as violins.
    
    Args:
        layer_match (dict): Dictionary of average percentage match values for each head in each layer.
        args (argparse.Namespace): Arguments containing at least 'model_path'.
    """
    # Prepare data for per-layer violin plot
    layers = []
    match_values = []

    for layer, values in layer_match.items():
        layers.extend([layer] * len(values))
        match_values.extend(values)

    # Save match values to a file
    trace_dir = "ablation_plots/traces/percdrift"
    os.makedirs(trace_dir, exist_ok=True)
    mpath = args.model_path.replace("/", "_")
    trace_path = os.path.join(trace_dir, f"decode_percdrift_{mpath}.npy")
    np.save(trace_path, {"Layer": layers, "Match": match_values})
    print(f"Decode percentage drift data saved to {trace_path}")

    # Plot the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=layers, y=match_values, scale="width", inner="quartile", palette="viridis")

    # Formatting
    plt.title(f"Per-Layer Percentage Match for {args.model_path}", fontsize=16)
    plt.xlabel("Layer", fontsize=14)
    plt.ylabel("Average Percentage Match (Top 10%)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()

    # Save the plot
    plot_path = f"ablation_plots/{mpath}_decode_percdrift_per_layer.pdf"
    plt.savefig(plot_path)
    print(f"Violin plot saved to {plot_path}")
    plt.close()


def plot_decode_drift_trajectory(decode_tokpos_affinity, top_k=0.1, args=None):
    """
    Plot the trajectory of top-k token overlaps for each decode step, compared to the first decode step.

    Args:
        decode_tokpos_affinity (dict): Dictionary where keys are layer indices and values are 
                                       torch.Tensor of shape [num_examples, num_heads, num_decode_steps, num_tokens].
        top_k (float): Percentage of top tokens to consider (e.g., 0.1 for top 10%).
        args (argparse.Namespace): Arguments containing at least 'model_path'.
    """


    trajectories = []  # To store trajectories for all layers and heads
    num_decode_steps = None  # To infer decode step count from the first layer
    for layer, tensor in tqdm(decode_tokpos_affinity.items(), desc="Processing Layers"):
        num_examples, num_heads, num_decode_steps, num_tokens = tensor.shape
        k = int(num_tokens * top_k)  # Top-k token count

        # Flatten examples and heads for batch processing
        decode_probs = tensor.view(-1, num_decode_steps, num_tokens)  # Shape: [163 * 24, 50, 974]

        # Get top-k indices for the first decode step
        initial_top_k = torch.topk(decode_probs[:, 0, :], k, dim=-1).indices  # Shape: [163 * 24, k]

        # Create binary masks for the first decode step
        initial_masks = torch.zeros(decode_probs.size(0), num_tokens, device=decode_probs.device)
        initial_masks.scatter_(1, initial_top_k, 1)  # Shape: [163 * 24, num_tokens]

        # Get top-k indices for all decode steps
        top_k_indices = torch.topk(decode_probs, k, dim=-1).indices  # Shape: [163 * 24, 50, k]

        # Create binary masks for all decode steps
        step_masks = torch.zeros(decode_probs.size(0), num_decode_steps, num_tokens, device=decode_probs.device)
        step_masks.scatter_(2, top_k_indices, 1)  # Shape: [163 * 24, 50, num_tokens]

        # Compute overlap with the first decode step for all steps
        overlaps = (step_masks * initial_masks.unsqueeze(1)).sum(dim=-1) / k  # Shape: [163 * 24, 50]

        # Append mean overlap trajectory for this layer
        # import pdb; pdb.set_trace()
        # Append all trajectories for this layer
        trajectories.extend(overlaps.cpu().numpy())  # Shape: [163 * 24, 50]

    # Plot all trajectories with a colormap
    # Convert trajectories to NumPy for easier processing
    trajectories = np.array(trajectories)  # Shape: [672, 50]
    plt.figure(figsize=(10, 6))
    colormap = cm.get_cmap("viridis", trajectories.shape[0])  # Viridis colormap
    for i, trajectory in enumerate(trajectories):
        plt.plot(range(num_decode_steps), trajectory, color=colormap(i), alpha=0.5, linewidth=0.8)

    # Add labels and grid
    plt.title("Decode Drift Trajectories for All Heads", fontsize=16)
    plt.xlabel("Decode Step", fontsize=14)
    plt.ylabel("Top-k Overlap with Initial Step", fontsize=14)
    plt.grid(True)

    # Save the plot
    mpath = args.model_path.replace("/", "_")
    output_path = f"ablation_plots/{mpath}_drift_trajectories.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
    bins = 10
    # Create Y-axis edges for histogram bins
    y_edges = np.linspace(0, 1, bins + 1)  # Divide Y-axis into bins (0 to 1 overlap values)

    # Digitize trajectories to find bin indices for all overlap values
    bin_indices = np.digitize(trajectories, y_edges, right=True) - 1  # Shape: [109536, 50]

    # Clip bin indices to avoid out-of-bound indices
    bin_indices = np.clip(bin_indices, 0, bins - 1)  # Ensure indices are within [0, bins-1]

    # Create the density map using bincount for each step
    density_map = np.zeros((bins, num_decode_steps), dtype=np.float32)
    for step in tqdm(range(num_decode_steps)):
        # Count occurrences in each bin for the current step
        counts = np.bincount(bin_indices[:, step], minlength=bins)
        density_map[:, step] = counts
    # Normalize density map for better visualization
    density_map /= density_map.max()
    # # Create a 2D density histogram (heatmap data)
    # density_map = np.zeros((bins, num_decode_steps))
    # y_edges = np.linspace(0, 1, bins + 1)  # Divide Y-axis into bins (0 to 1 overlap values)

    # for step in range(num_decode_steps):
    #     # Histogram for overlap values at each decode step
    #     hist, _ = np.histogram(trajectories[:, step], bins=y_edges)
    #     density_map[:, step] = hist

    # # Normalize density map for better visualization
    # density_map /= density_map.max()

    # Plot the density heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        density_map,
        cmap="viridis",
        xticklabels=10,  # Optional: Adjust frequency of X-ticks
        yticklabels=np.round(np.linspace(0, 1, bins), decimals=2),  # Show overlap bins
        cbar_kws={"label": "Density"}
    )
    plt.title("Density of Decode Drift Trajectories", fontsize=16)
    plt.xlabel("Decode Step", fontsize=14)
    plt.ylabel("Top-k Overlap with Initial Step", fontsize=14)
    plt.tight_layout()

    # Save the plot
    mpath = args.model_path.replace("/", "_")
    output_path = f"ablation_plots/{mpath}_drift_density_heatmap.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=600)
    plt.close()

    print(f"Drift trajectory plot saved to {output_path}")
    # Convert trajectories to NumPy for easier processing
    trajectories = np.array(trajectories)  # Shape: [672, 50]
    # average trajectories on dim=0
    trajectories_to_save = np.mean(trajectories, axis=0)  # Shape: [50]

    # Save rank agreement values to a .npy file
    trace_dir = "ablation_plots/traces/decode_drift_trajectory"
    os.makedirs(trace_dir, exist_ok=True)
    mpath = args.model_path.replace("/", "_")
    trace_path = os.path.join(trace_dir, f"drift_traj_{mpath}.npy")
    np.save(trace_path, {"Trajectory": trajectories_to_save})

    # Compute mean and standard deviation across trajectories
    mean_trajectory = np.mean(trajectories, axis=0)  # Shape: [50]
    std_trajectory = np.std(trajectories, axis=0)  # Shape: [50]

    # Plot the mean trajectory with shaded error region
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_decode_steps), mean_trajectory, label="Mean Drift Trajectory", color="blue")
    plt.fill_between(
        range(num_decode_steps),
        mean_trajectory - std_trajectory,
        mean_trajectory + std_trajectory,
        color="blue",
        alpha=0.2,
        label="±1 Std Dev"
    )
    plt.axhline(y=1.0, color="red", linestyle="--", label="Initial (100% Overlap)")
    plt.title("Decode Drift Trajectory with Error Region", fontsize=16)
    plt.xlabel("Decode Step", fontsize=14)
    plt.ylabel("Top-k Overlap with Initial Step", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Save the plot
    mpath = args.model_path.replace("/", "_")
    output_path = f"ablation_plots/{mpath}_drift_trajectory.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()

    print(f"Drift trajectory plot saved to {output_path}")


def compute_rank_agreement_all_examples(head_tokpos_affinity, args):
    """
    Compute rank agreement (mean, min, max) for all examples across all heads and layers.

    Args:
        head_tokpos_affinity (dict): Keys are layers; values are torch.Tensor of shape [num_examples, num_heads, num_tokens].

    Returns:
        np.ndarray: Shape [num_examples, 3], where 3 corresponds to mean, min, and max rank correlation per example.
    """
    num_examples = next(iter(head_tokpos_affinity.values())).shape[0]

    # Flatten heads across layers
    all_heads = []
    for layer, tensor in head_tokpos_affinity.items():
        all_heads.append(tensor)  # Shape: [num_examples, num_heads, num_tokens]
    all_heads = torch.cat(all_heads, dim=1)  # Shape: [num_examples, total_heads, num_tokens]

    # Rank tokens per head (Spearman's correlation requires ranks)
    ranks = torch.argsort(all_heads, dim=-1).float()  # Shape: [num_examples, total_heads, num_tokens]

    # Compute rank correlation metrics
    results = []
    total_corr_matrix = None

    for example_idx in tqdm(range(num_examples), desc="Computing Rank Correlations"):
        example_ranks = ranks[example_idx]  # Shape: [total_heads, num_tokens]

        # Compute rank correlation matrix (Spearman)
        corr_matrix = np.corrcoef(example_ranks.numpy())  # Shape: [total_heads, total_heads]

        # Accumulate the correlation matrix
        if total_corr_matrix is None:
            total_corr_matrix = corr_matrix
        else:
            total_corr_matrix += corr_matrix
        # Extract upper triangle
        triu_indices = np.triu_indices(corr_matrix.shape[0], k=1)
        upper_triangle = corr_matrix[triu_indices]

        # Compute mean, min, and max of rank correlations
        mean_corr = np.mean(upper_triangle)
        min_corr = np.min(upper_triangle)
        max_corr = np.max(upper_triangle)

        results.append([mean_corr, min_corr, max_corr])

    mean_corr_matrix = total_corr_matrix / num_examples
    # Plot the heatmap
    plt.figure(figsize=(12, 10))  # Adjust size as needed
    sns.heatmap(corr_matrix, square=True, cbar=True, xticklabels=False, yticklabels=False, cmap="viridis")

    # Set title
    plt.title("Mean Rank Correlation Matrix", fontsize=16)

    # Construct the file path
    mpath = args.model_path.replace("/", "_")
    heatmap_path = f"ablation_plots/{mpath}_rankcorr_heatmap.png"

    # Save the heatmap
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=600)
    plt.close()

    print(f"Mean rank correlation heatmap saved to {heatmap_path}")
    return np.array(results)  # Shape: [num_examples, 3]

def plot_and_save_rank_agreement(rank_agreement, args):
    """
    Save rank agreement values and plot their distribution as a violin plot.

    Args:
        rank_agreement (np.ndarray): Shape [num_examples, 3], where columns represent mean, min, and max rank correlations.
        args (argparse.Namespace): Arguments containing at least 'model_path'.
    """
    # Save rank agreement values to a .npy file
    trace_dir = "ablation_plots/traces/rankagreement_allheads"
    os.makedirs(trace_dir, exist_ok=True)
    mpath = args.model_path.replace("/", "_")
    trace_path = os.path.join(trace_dir, f"rank_agreement_{mpath}.npy")
    np.save(trace_path, {"RankAgreement": rank_agreement})
    print(f"Rank agreement data saved to {trace_path}")

    # Prepare data for violin plot
    categories = ["Mean", "Min", "Max"]
    values = [rank_agreement[:, i] for i in range(3)]  # Separate columns

    # Create the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=values, scale="width", inner="quartile", palette="viridis")

    # Formatting
    plt.title(f"Rank Agreement Distribution for {mpath}", fontsize=16)
    plt.xlabel("Metric", fontsize=14)
    plt.ylabel("Rank Correlation", fontsize=14)
    plt.xticks(range(3), categories, fontsize=12)
    plt.yticks(fontsize=12)

    # Enhance layout and save the plot
    plt.tight_layout()
    # plot_path = os.path.join(trace_dir, f"rank_agreement_violin_{mpath}.pdf")

    plot_path = f"ablation_plots/{mpath}_rank_agreement_violin.pdf"
    plt.savefig(plot_path)
    print(f"Violin plot saved to {plot_path}")
    plt.close()
    