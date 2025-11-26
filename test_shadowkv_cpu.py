import torch
import sys
import os
from types import SimpleNamespace

# Add current directory to sys.path
sys.path.append(os.getcwd())

from models.kv_cache import ShadowKVCache_CPU

config = SimpleNamespace(
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=8,
    hidden_size=128,
)

try:
    kv_cache = ShadowKVCache_CPU(
        config=config,
        batch_size=1,
        max_length=128,
        device='cpu',
        dtype=torch.float32,
        sparse_budget=64,
        chunk_size=8,
        rank=16
    )
    print("ShadowKVCache_CPU instantiated successfully")
    kv_cache.print_stats()
except Exception as e:
    print(f"Failed to instantiate ShadowKVCache_CPU: {e}")
