################################################################################
#
# TokenButler Predictor Integration for xKV
# 
# This module ports the TokenImportancePredictorAttentive from TokenButler
# for use with the xKV sparse attention system.
#
################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class TokenButlerPredictor(nn.Module):
    """
    Token importance predictor that learns to predict which KV cache tokens 
    are most relevant for a given query, enabling sparse attention during decode.
    
    This predictor uses a producer/consumer pattern where one predictor serves
    multiple consecutive layers (controlled by producer_frequency).
    
    Architecture:
        - Input: Hidden states from transformer [B, L, hidden_size]
        - Output: Importance queries [B*H, N_slots, Lq, dDash]
        - K comes from real key cache via learned projection
    """
    
    def __init__(
        self,
        config,
        pred_hid_size: int,
        num_heads: int,
        num_hidden_layers: int,  # N_slots = producer_frequency
        dDash: int = 16,
        intermediate_dim: int = 1024,
        attn_reduce_factor: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            config: Model configuration object with rope_theta, max_position_embeddings, etc.
            pred_hid_size: Hidden size for predictor (typically same as model hidden_size)
            num_heads: Number of attention heads
            num_hidden_layers: Number of layers this predictor serves (N_slots)
            dDash: Reduced dimension for importance computation
            intermediate_dim: MLP intermediate dimension
            attn_reduce_factor: Factor to reduce hidden size for attention
            dropout: Dropout probability
        """
        super().__init__()
        self.config = config
        self.hidden_size = pred_hid_size
        self.num_heads = num_heads
        # NOTE: this arg is actually N_slots == producer_frequency (not total transformer layers)
        self.num_hidden_layers = num_hidden_layers  # N_slots
        self.dropout = dropout
        self.dDash = dDash
        self.intermediate_dim = intermediate_dim
        self.attn_reduce_factor = attn_reduce_factor
        
        # Real model head dim (for projecting true K cache)
        num_attn_heads = getattr(config, "num_attention_heads", num_heads)
        self.model_head_dim = config.hidden_size // num_attn_heads
        
        # Validate dDash
        if self.dDash > self.model_head_dim:
            raise ValueError(
                f"dDash={self.dDash} must be <= model head dim={self.model_head_dim}"
            )
        
        # Number of KV heads (for GQA models, this is less than num_attention_heads)
        self.num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
        self.num_key_value_groups = num_heads // self.num_key_value_heads

        # --- IMPORTANT ---
        # Baseline TokenButler has one predictor *per producer layer* (layers 0, G, 2G, ...),
        # i.e. Q-MLP weights are *not* shared across producers.
        # Your previous xKV port loaded only producer 0 weights → large accuracy drop.
        self.producer_frequency = self.num_hidden_layers
        total_layers = getattr(config, "num_hidden_layers", 32)
        self.num_producers = math.ceil(total_layers / self.producer_frequency)

        # Per-producer LayerNorm + Q-MLP
        self.norm_importance = nn.ModuleList(
            [nn.LayerNorm(self.hidden_size) for _ in range(self.num_producers)]
        )

        # Q-MLP: predicts queries for N_slots layers
        # Output: [B, L, N_slots * H * dDash] (H = num_attention_heads for Q)
        out_dim = self.num_hidden_layers * self.num_heads * self.dDash

        self.q_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(pred_hid_size, self.intermediate_dim, bias=False),
                nn.SiLU(),
                nn.Linear(self.intermediate_dim, out_dim, bias=False),
            )
            for _ in range(self.num_producers)
        ])

        # Per-(layer, kv_head) projection of real KV cache keys
        # Shape: [num_hidden_layers (total), num_key_value_heads, head_dim, dDash]
        # We store per KV head (8) not per attention head (32) to save 4x memory
        # During loading, we aggregate per-attention-head projections to per-KV-head
        self.key_cache_proj = nn.Parameter(
            torch.empty(total_layers, self.num_key_value_heads, self.model_head_dim, self.dDash)
        )
        nn.init.xavier_uniform_(self.key_cache_proj.view(-1, self.dDash))
        
        self._initialize_weights()
        self.device = None
    
    def _initialize_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        producer_layer_idx: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass for importance query prediction.
        
        Args:
            hidden_states: [B, L, hidden_size] - transformer hidden states
            attention_mask: Optional attention mask (not used in current implementation)
            position_ids: Optional position IDs (not used in current implementation)
            producer_layer_idx: Index of the producer layer (for multi-producer setups)
            
        Returns:
            q_importance: [B*H, N_slots, L, dDash] - predicted importance queries
        """
        if self.device != hidden_states.device:
            self.device = hidden_states.device
            self.to(self.device)
        
        B, L, _ = hidden_states.size()
        H = self.num_heads # This is the number of attention heads
        N_slots = self.num_hidden_layers

        # Select producer id based on producer_layer_idx (0, G, 2G, ...)
        # Clamp for safety in case of odd configs.
        prod_id = int(producer_layer_idx // self.producer_frequency)
        prod_id = max(0, min(prod_id, self.num_producers - 1))

        mlp = self.q_mlp[prod_id]
        norm = self.norm_importance[prod_id]

        # Normalize and project through MLP
        hidden_states = hidden_states.to(mlp[0].weight.dtype)
        hidden_for_importance = norm(hidden_states)

        # MLP output: [B, L, N_slots * H * dDash]
        q_flat = mlp(hidden_for_importance)
        
        # Reshape: [B, L, N_slots, H, dDash]
        q_slot = q_flat.view(B, L, N_slots, H, self.dDash)

        # Permute to [B, H, N_slots, L, dDash]
        # OPTIMIZATION: Remove .contiguous() to avoid expensive memory copy
        # PyTorch can handle non-contiguous tensors for indexing operations
        q_slot = q_slot.permute(0, 3, 2, 1, 4)

        # Final shape: [B*H, N_slots, L, dDash]
        # Use reshape instead of view - it handles non-contiguous tensors automatically
        q_importance = q_slot.reshape(B * H, N_slots, L, self.dDash)
        
        return q_importance
    
    def project_key_cache(
        self,
        key_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Project key cache to reduced dimension for importance computation.
        
        Args:
            key_states: [B, num_kv_heads, L, head_dim] - key cache states (per KV head)
            layer_idx: Layer index for selecting projection weights
            
        Returns:
            k_proj: [B, num_kv_heads, L, dDash] - projected keys (per KV head)
        """
        # proj_weight: [num_kv_heads, head_dim, dDash]
        proj_weight = self.key_cache_proj[layer_idx]
        
        # key_states: [B, num_kv_heads, L, head_dim]
        # Einsum: "bhlk,hkd->bhld"
        k_proj = torch.einsum("bhlk,hkd->bhld", key_states, proj_weight)
        
        return k_proj
    
    def compute_importance_scores(
        self,
        q_importance: torch.Tensor,
        k_proj: torch.Tensor,
        slot_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute importance scores between query importance and projected keys.
        
        Args:
            q_importance: [B*H, N_slots, Lq, dDash] - importance queries
            k_proj: [B*H, Lk, dDash] - projected keys (flattened batch*heads)
            slot_idx: Which slot to use from q_importance
            attention_mask: Optional mask to apply
            
        Returns:
            importance_scores: [B*H, Lq, Lk] - attention-like importance scores
        """
        # Select the slot: [B*H, Lq, dDash]
        q_slot = q_importance[:, slot_idx, :, :]
        
        # Compute scores: [B*H, Lq, Lk]
        scores = torch.bmm(q_slot, k_proj.transpose(-2, -1)) / math.sqrt(self.dDash)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax over key dimension
        importance_scores = F.softmax(scores, dim=-1)
        
        return importance_scores


def load_tokenbutler_predictor(
    config,
    predictor_path: str,
    num_heads: int,
    producer_frequency: int = 4,
    dDash: int = 32,
    intermediate_dim: int = 512,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> TokenButlerPredictor:
    """
    Load a trained TokenButler predictor from checkpoint.
    
    The checkpoint format from TokenButler training is:
    {
        'model_state_dict': [  # List of state_dicts, one per producer layer
            {  # Producer layer 0 (serves layers 0, 1, 2, 3)
                'sparse_token_predictor.q_mlps.0.0.weight': ...,
                'sparse_token_predictor.q_mlps.0.2.weight': ...,
                'sparse_token_predictor.key_cache_proj': [num_layers, H, head_dim, dDash],
                'sparse_token_predictor.norm_importance.weight': ...,
                'sparse_token_predictor.norm_importance.bias': ...,
            },
            {  # Producer layer 1 (serves layers 4, 5, 6, 7)
                ...
            },
            ...
        ]
    }
    
    Args:
        config: Model configuration
        predictor_path: Path to predictor weights (.pt file)
        num_heads: Number of attention heads
        producer_frequency: Number of layers served by one predictor
        dDash: Reduced dimension for importance
        intermediate_dim: MLP intermediate dimension
        device: Device to load to
        dtype: Data type
        
    Returns:
        Initialized TokenButlerPredictor with loaded weights
    """
    predictor = TokenButlerPredictor(
        config=config,
        pred_hid_size=config.hidden_size,
        num_heads=num_heads,
        num_hidden_layers=producer_frequency,
        dDash=dDash,
        intermediate_dim=intermediate_dim,
    )
    
    if predictor_path and predictor_path != "":
        checkpoint = torch.load(predictor_path, map_location="cpu", weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict_list = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, list):
            state_dict_list = checkpoint
        else:
            # Assume it's a direct state_dict
            predictor.load_state_dict(checkpoint, strict=False)
            print(f"Loaded TokenButler predictor from {predictor_path}")
            predictor = predictor.to(device).to(dtype)
            predictor.eval()
            return predictor
        
        # state_dict_list is a list of state_dicts, one per producer layer.
        # We merge:
        #   - per-producer Q-MLP + norm weights (NOT shared)
        #   - per-layer key_cache_proj rows from the *correct* producer (shifted mapping)
        print(f"Found {len(state_dict_list)} producer layer weights in checkpoint")

        mapped_state_dict: dict[str, torch.Tensor] = {}

        for prod_idx, prod_sd in enumerate(state_dict_list):
            for key, value in prod_sd.items():
                k = key
                if k.startswith("sparse_token_predictor."):
                    k = k.replace("sparse_token_predictor.", "")

                # Per-producer Q-MLP:
                # baseline key: q_mlps.0.0.weight / q_mlps.0.2.weight
                # ours:         q_mlp.{prod_idx}.0.weight / q_mlp.{prod_idx}.2.weight
                if k.startswith("q_mlps.0."):
                    mapped_state_dict[k.replace("q_mlps.0.", f"q_mlp.{prod_idx}.")] = value
                    continue

                # Per-producer LayerNorm:
                if k == "norm_importance.weight":
                    mapped_state_dict[f"norm_importance.{prod_idx}.weight"] = value
                    continue
                if k == "norm_importance.bias":
                    mapped_state_dict[f"norm_importance.{prod_idx}.bias"] = value
                    continue

                # key_cache_proj handled separately (merged by correct producer mapping)
                if k.endswith("key_cache_proj"):
                    continue

        missing, unexpected = predictor.load_state_dict(mapped_state_dict, strict=False)
        print(f"Loaded per-producer Q-MLP+norm weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing keys: {missing[:5]}..." if len(missing) > 5 else f"  Missing keys: {missing}")
        
        def _extract_key_cache_proj(sd: dict) -> Optional[torch.Tensor]:
            candidates: list[tuple[str, torch.Tensor]] = []
            for k, v in sd.items():
                if k.endswith("key_cache_proj"):
                    candidates.append((k, v))
            if not candidates:
                return None

            # Prefer the least-prefixed key (closest to the base module).
            # Example preferred order:
            #   sparse_token_predictor.key_cache_proj
            #   producer.sparse_token_predictor.key_cache_proj
            #   producer.producer.sparse_token_predictor.key_cache_proj
            def _rank_key(item: tuple[str, torch.Tensor]) -> tuple[int, int]:
                k, _ = item
                return (k.count("producer."), len(k))

            candidates.sort(key=_rank_key)
            return candidates[0][1]

        sample_key_proj = None
        for sd in state_dict_list:
            sample_key_proj = _extract_key_cache_proj(sd)
            if sample_key_proj is not None:
                break

        is_already_gqa = False
        num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
        total_layers = config.num_hidden_layers

        if sample_key_proj is not None and sample_key_proj.shape[1] == num_key_value_heads:
            is_already_gqa = True
            print(f"  Checkpoint has GQA shape ({num_key_value_heads} heads). Loading directly.")

        # --- IMPORTANT: correct producer→layer mapping (matches baseline) ---
        # Baseline mapping for consumer layers:
        #   producer at p serves layers (p+1 ... p+G)
        # So the producer index for a given layer L>0 is: (L-1)//G.
        # (Layer 0 doesn't use key_cache_proj for sparsity; we still fill it from producer 0.)
        if is_already_gqa:
            merged_key_proj = torch.zeros(
                total_layers,
                num_key_value_heads,
                predictor.model_head_dim,
                dDash,
            )
            for layer_idx in range(total_layers):
                prod_idx = 0 if layer_idx == 0 else (layer_idx - 1) // producer_frequency
                prod_idx = min(prod_idx, len(state_dict_list) - 1)
                key_proj = _extract_key_cache_proj(state_dict_list[prod_idx])
                if key_proj is None:
                    continue
                merged_key_proj[layer_idx] = key_proj[layer_idx]
        else:
            merged_key_proj_full = torch.zeros(
                total_layers,
                num_heads,
                predictor.model_head_dim,
                dDash,
            )
            for layer_idx in range(total_layers):
                prod_idx = 0 if layer_idx == 0 else (layer_idx - 1) // producer_frequency
                prod_idx = min(prod_idx, len(state_dict_list) - 1)
                key_proj = _extract_key_cache_proj(state_dict_list[prod_idx])
                if key_proj is None:
                    continue
                merged_key_proj_full[layer_idx] = key_proj[layer_idx]

            num_key_value_groups = num_heads // num_key_value_heads
            merged_key_proj = merged_key_proj_full.view(
                total_layers,
                num_key_value_heads,
                num_key_value_groups,
                predictor.model_head_dim,
                dDash,
            ).mean(dim=2)
            print(f"  Aggregated key_cache_proj: {num_heads} attn heads -> {num_key_value_heads} KV heads")
        
        # Assign aggregated key_cache_proj
        predictor.key_cache_proj.data.copy_(merged_key_proj)
        print(f"Loaded TokenButler predictor from {predictor_path}")
    else:
        print("Initialized TokenButler predictor with random weights")
    
    predictor = predictor.to(device).to(dtype)
    predictor.eval()
    
    return predictor
