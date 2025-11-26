import torch
import sys
import os

# Add current directory to sys.path
sys.path.append(os.getcwd())

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

try:
    from kernels import shadowkv
    print("ShadowKV imported successfully from kernels")
except ImportError as e:
    print(f"Failed to import shadowkv: {e}")
