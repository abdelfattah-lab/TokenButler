import torch
from transformers.cache_utils import DynamicCache
import gc

from transformers.models.mistral.modeling_mistral import (
    apply_rotary_pos_emb,
)

from ..configurations import xKVConfig

def fake_svd(tensor, rank):
    """Perform fake SVD: SVD -> Truncate -> Multiply back."""
    bs, nh, sl, hd = tensor.shape
    tensor_reshaped = tensor.transpose(1, 2).reshape(bs, sl, nh * hd)
    
    # Step 1: Perform SVD NOTE(brian1009): Have deterministic issue but faster
    #U_trunc, S_trunc, V_trunc = torch.svd_lowrank(tensor_reshaped, q=rank)
    #Vt_trunc = V_trunc.transpose(1, 2)
    
    U, S, V_h = torch.linalg.svd(tensor_reshaped, full_matrices=False)
    U_trunc = U[:, :, :rank]
    S_trunc = S[:, :rank]
    Vt_trunc = V_h[:, :rank, :]
    
    # Step 2: Multiply back to approximate the original tensor
    approx_tensor = torch.matmul(U_trunc, torch.matmul(torch.diag_embed(S_trunc), Vt_trunc)) # (bs, sl, nh * hd)
    approx_tensor = approx_tensor.view(bs, sl, nh, hd).transpose(1, 2)
    
    return approx_tensor


class FakeLayerMergingCache(DynamicCache):
    def __init__(
        self,
        merge_setup: xKVConfig,
    ):
        """Simplified interface: num_heads and head_dim are inferred from the input tensors."""
        super().__init__()
        self.num_layers = merge_setup.num_layers
        self.merge_setup = merge_setup

    def _should_merge(self, layer_idx):
        """Check if this layer is the last in its merge group using dictionary lookup."""
        group_info = self.merge_setup.get_group_for_layer(layer_idx)
        if group_info is not None: # No group found
            last_layer_idx_in_group = group_info.layers[-1]
            return layer_idx == last_layer_idx_in_group
        return False
        
    def is_value_merged(self):
        return self.merge_setup.merge_value

    def is_key_merged(self):
        return self.merge_setup.merge_key

    def update(self, key, value, layer_idx, mode='prefill', cos=None, sin=None, re_apply_rope=True):
        """Override update to hook into the fake SVD compression process."""
        super().update(key, value, layer_idx)

        if mode == 'prefill':
            # Infer num_heads and head_dim from the key shape
            self.num_heads = key.shape[1]  # Shape: (batch_size, num_heads, seq_len, head_dim)
            self.head_dim = key.shape[3]

            # Apply grouped fake SVD if we have updated the last layer in the group
            if self._should_merge(layer_idx):
                self.grouped_layer_merging(layer_idx)
            
            group_info = self.merge_setup.get_group_for_layer(layer_idx)
            if group_info is not None: # grouped founded      
                if layer_idx == group_info.layers[-1]: # last layer in the group
                    for layer in group_info.layers:
                        pre_rope_key = self.key_cache[layer]
                        if re_apply_rope:
                            _, self.key_cache[layer] = apply_rotary_pos_emb(pre_rope_key, pre_rope_key, cos, sin)
            else: # no group found  
                pre_rope_key = self.key_cache[layer_idx]
                if re_apply_rope:
                    _, self.key_cache[layer_idx] = apply_rotary_pos_emb(pre_rope_key, pre_rope_key, cos, sin)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    @torch.no_grad()
    def grouped_layer_merging(self, last_layer_idx):
        """Perform fake SVD on grouped layers, inferring dimensions from the tensors."""
        group_info = self.merge_setup.get_group_for_layer(last_layer_idx)
        if group_info is None:
            return  # No valid group found
        start_layer_idx, end_layer_idx = group_info.layers[0], group_info.layers[-1]       

        # Step 1: Collect keys and values for the layers in the group            
        keys, values = zip(*[self.__getitem__(i) for i in range(start_layer_idx, end_layer_idx + 1)])
        split_sizes = [self.num_heads for _ in range(start_layer_idx, end_layer_idx + 1)]
        
        if self.merge_setup.layer_merge_impl == 'svd':
            # Step 2: Concatenate along the sequence length dimension
            combined_key = torch.cat(keys, dim=1)  # Shape: (batch_size, total_num_heads, seq_len, head_dim)
            combined_value = torch.cat(values, dim=1)

            # Step 3: Apply fake SVD (truncate and multiply back)
            #NOTE(brian1009): Experiment with fake SVD on key only for now
            if self.merge_setup.merge_key:
                combined_key = fake_svd(combined_key.float(), rank=group_info.rank_k).to(combined_key.dtype)
            if self.merge_setup.merge_value:
                combined_value = fake_svd(combined_value.float(), rank=group_info.rank_v).to(combined_value.dtype)

            # Step 4: Split and update the cache for each layer
            key_layers = torch.split(combined_key, split_sizes, dim=1)
            value_layers = torch.split(combined_value, split_sizes, dim=1)
        else:
            raise NotImplementedError(f"Unknown implementation: {self.impl}")
        
        for idx, layer_idx in enumerate(range(start_layer_idx, end_layer_idx + 1)):
            self.update_cache(layer_idx, key_layers[idx], value_layers[idx])

        # TODO(max410011): Uncomment these lines during memory usage evaluation
        # torch.cuda.synchronize()
        # gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        
    def update_cache(self, layer_idx, key_approx, value_approx):
        """Update the cache with the approximated key and value tensors."""
        self.key_cache[layer_idx] = key_approx
        self.value_cache[layer_idx] = value_approx
    
    def get_max_length(self):
        # BC for DeepSeek-Coder-V2-Lite-Instruct
        return None