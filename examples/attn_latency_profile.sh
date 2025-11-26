# Baseline
nsys profile -t cuda,nvtx,osrt --stats=true --force-overwrite=true \
    -o baseline_bs4 python test/e2e.py --bsz 4 --gen_len 100 --baseline

# ShadowKV-k64
nsys profile -t cuda,nvtx,osrt --stats=true --force-overwrite=true \
    -o shadowkv_k64_bs4 python test/e2e.py --bsz 4 --gen_len 100 --budget 2048 --shadowkv --rank_k 64

# ShadowKV-k128
nsys profile -t cuda,nvtx,osrt --stats=true --force-overwrite=true \
    -o shadowkv_k128_bs4 python test/e2e.py --bsz 4 --gen_len 100 --budget 2048 --shadowkv --rank_k 128

# ShadowKV-xKey-1-k64
nsys profile -t cuda,nvtx,osrt --stats=true --force-overwrite=true \
    -o xKey-1_k64_bs4 python test/e2e.py --bsz 4 --gen_len 100 --budget 2048 --xkey --group_size 1 --rank_k 64

# ShadowKV-xKey-2-k128
nsys profile -t cuda,nvtx,osrt --stats=true --force-overwrite=true\
    -o xKey-2_k128_bs4 python test/e2e.py --bsz 4 --gen_len 100 --budget 2048 --xkey --group_size 2 --rank_k 128

# ShadowKV-xKey-4-k256
nsys profile -t cuda,nvtx,osrt --stats=true --force-overwrite=true\
    -o xKey-4_k256_bs4 python test/e2e.py --bsz 4 --gen_len 100 --budget 2048 --xkey --group_size 4 --rank_k 256


nsys stats --report cuda_gpu_kern_sum --format csv baseline_bs4.nsys-rep -o baseline_bs4
nsys stats --report cuda_gpu_kern_sum --format csv shadowkv_k64_bs4.nsys-rep -o shadowkv_k64_bs4
nsys stats --report cuda_gpu_kern_sum --format csv shadowkv_k128_bs4.nsys-rep -o shadowkv_k128_bs4
nsys stats --report cuda_gpu_kern_sum --format csv xKey-1_k64_bs4.nsys-rep -o xKey-1_k64_bs4
nsys stats --report cuda_gpu_kern_sum --format csv xKey-2_k128_bs4.nsys-rep -o xKey-2_k128_bs4
nsys stats --report cuda_gpu_kern_sum --format csv xKey-4_k256_bs4.nsys-rep -o xKey-4_k256_bs4

python3 scripts/extract_nsys_kernel_times.py --path . --out kernel_times.csv