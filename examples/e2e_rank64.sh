# ---------------------- e2e throughput test - all 65536 -------------------
# rank 64, bs 1
python test/e2e.py --datalen 65536 --bsz 1 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs1.log

# rank 64, bs 2
python test/e2e.py --datalen 65536 --bsz 2 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs2.log

# rank 64, bs 4
python test/e2e.py --datalen 65536 --bsz 4 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs4.log

# rank 64, bs 8
python test/e2e.py --datalen 65536 --bsz 8 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs8.log

# rank 64, bs 16
python test/e2e.py --datalen 65536 --bsz 16 --gen_len 100 --budget 2048 --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs16.log

# rank 64, bs 32
python test/e2e.py --datalen 65536 --bsz 32 --gen_len 100 --budget 2048 --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs32.log


# ---------------------- e2e throughput test - all 131072 -------------------
# rank 64, bs 1
python test/e2e.py --datalen 131072 --bsz 1 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs1.log

# rank 64, bs 2
python test/e2e.py --datalen 131072 --bsz 2 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs2.log

# rank 64, bs 4
python test/e2e.py --datalen 131072 --bsz 4 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs4.log

# rank 64, bs 8
python test/e2e.py --datalen 131072 --bsz 8 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs8.log

# rank 64, bs 16
python test/e2e.py --datalen 131072 --bsz 16 --gen_len 100 --budget 2048 --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs16.log

# rank 64, bs 32
python test/e2e.py --datalen 131072 --bsz 32 --gen_len 100 --budget 2048 --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs32.log



# ---------------------- e2e throughput test - all 262144 -------------------
# rank 64, bs 1
python test/e2e.py --datalen 262144 --bsz 1 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs1.log

# rank 64, bs 2
python test/e2e.py --datalen 262144 --bsz 2 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs2.log

# rank 64, bs 4
python test/e2e.py --datalen 262144 --bsz 4 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs4.log

# rank 64, bs 8
python test/e2e.py --datalen 262144 --bsz 8 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs8.log

# rank 64, bs 16
python test/e2e.py --datalen 262144 --bsz 16 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs16.log

# rank 64, bs 32
python test/e2e.py --datalen 262144 --bsz 32 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs32.log


# ---------------------- e2e throughput test - all 524288 -------------------
# rank 64, bs 1
python test/e2e.py --datalen 524288 --bsz 1 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs1.log

# rank 64, bs 2
python test/e2e.py --datalen 524288 --bsz 2 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs2.log

# rank 64, bs 4
python test/e2e.py --datalen 524288 --bsz 4 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs4.log

# rank 64, bs 8
python test/e2e.py --datalen 524288 --bsz 8 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs8.log

# rank 64, bs 16
python test/e2e.py --datalen 524288 --bsz 16 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 64 --xkey --group_size 1 2 4 --rank_k 64 128 256 | tee -a logs/rank64_bs16.log




python test/e2e.py --datalen 65536 --bsz 4 --gen_len 100 --budget 2048 --baseline --xkey --group_size 4 --rank_k 256
