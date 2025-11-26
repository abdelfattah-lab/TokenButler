import torch, time

size = 1024*1024*256  # 256 KB
# Allocate host tensor in pinned memory for faster H2D/D2H transfers.
h = torch.randn(size, device='cpu', pin_memory=True)
d = torch.empty_like(h, device='cuda')

# Host → Device
torch.cuda.synchronize()
start = time.time()
d.copy_(h, non_blocking=True)
torch.cuda.synchronize()
h2d_bandwidth = size*4 / 1e9 / (time.time()-start)
print(f"H2D: {h2d_bandwidth:.2f} GB/s")

# Device → Host
torch.cuda.synchronize()
start = time.time()
h.copy_(d, non_blocking=True)
torch.cuda.synchronize()
d2h_bandwidth = size*4 / 1e9 / (time.time()-start)
print(f"D2H: {d2h_bandwidth:.2f} GB/s")
