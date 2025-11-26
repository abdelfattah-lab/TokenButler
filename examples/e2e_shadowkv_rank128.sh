# ---------------------- e2e throughput test - shadowkv 65536 -------------------
# rank 128, bs 1
python test/e2e.py --datalen 65536 --bsz 1 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 128 | tee -a logs/rank128_bs1.log

# rank 128, bs 2
python test/e2e.py --datalen 65536 --bsz 2 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 128 | tee -a logs/rank128_bs2.log

# rank 128, bs 4
python test/e2e.py --datalen 65536 --bsz 4 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 128 | tee -a logs/rank128_bs4.log

# rank 128, bs 8
python test/e2e.py --datalen 65536 --bsz 8 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 128 | tee -a logs/rank128_bs8.log

# rank 128, bs 16
python test/e2e.py --datalen 65536 --bsz 16 --gen_len 100 --budget 2048 --shadowkv --rank 128 | tee -a logs/rank128_bs16.log

# rank 128, bs 32
python test/e2e.py --datalen 65536 --bsz 32 --gen_len 100 --budget 2048 --shadowkv --rank 128 | tee -a logs/rank128_bs32.log

# rank 128, bs 64
python test/e2e.py --datalen 65536 --bsz 64 --gen_len 100 --budget 2048 --shadowkv --rank 128 | tee -a logs/rank128_bs64.log


# ---------------------- e2e throughput test - shadowkv 131072 -------------------
# rank 128, bs 1
python test/e2e.py --datalen 131072 --bsz 1 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 128 | tee -a logs/rank128_bs1.log

# rank 128, bs 2
python test/e2e.py --datalen 131072 --bsz 2 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 128 | tee -a logs/rank128_bs2.log

# rank 128, bs 4
python test/e2e.py --datalen 131072 --bsz 4 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 128 | tee -a logs/rank128_bs4.log

# rank 128, bs 8
python test/e2e.py --datalen 131072 --bsz 8 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 128 | tee -a logs/rank128_bs8.log

# rank 128, bs 16
python test/e2e.py --datalen 131072 --bsz 16 --gen_len 100 --budget 2048 --shadowkv --rank 128 | tee -a logs/rank128_bs16.log

# rank 128, bs 32
python test/e2e.py --datalen 131072 --bsz 32 --gen_len 100 --budget 2048 --shadowkv --rank 128 | tee -a logs/rank128_bs32.log



# ---------------------- e2e throughput test - shadowkv 262144 -------------------
# rank 128, bs 1
python test/e2e.py --datalen 262144 --bsz 1 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 128 | tee -a logs/rank128_bs1.log

# rank 128, bs 2
python test/e2e.py --datalen 262144 --bsz 2 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 128 | tee -a logs/rank128_bs2.log

# rank 128, bs 4
python test/e2e.py --datalen 262144 --bsz 4 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 128 | tee -a logs/rank128_bs4.log

# rank 128, bs 8
python test/e2e.py --datalen 262144 --bsz 8 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 128 | tee -a logs/rank128_bs8.log

# rank 128, bs 16
python test/e2e.py --datalen 262144 --bsz 16 --gen_len 100 --budget 2048 --shadowkv --rank 128 | tee -a logs/rank128_bs16.log

# rank 128, bs 32
python test/e2e.py --datalen 262144 --bsz 32 --gen_len 100 --budget 2048 --shadowkv --rank 128 | tee -a logs/rank128_bs32.log



# ---------------------- e2e throughput test - shadowkv 524288 -------------------
# rank 128, bs 1
python test/e2e.py --datalen 524288 --bsz 1 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 128 | tee -a logs/rank128_bs1.log

# rank 128, bs 2
python test/e2e.py --datalen 524288 --bsz 2 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 128 | tee -a logs/rank128_bs2.log

# rank 128, bs 4
python test/e2e.py --datalen 524288 --bsz 4 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 128 | tee -a logs/rank128_bs4.log

# rank 128, bs 8
python test/e2e.py --datalen 524288 --bsz 8 --gen_len 100 --budget 2048 --baseline --shadowkv --rank 128 | tee -a logs/rank128_bs8.log
