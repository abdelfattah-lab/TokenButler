#!/usr/bin/env python
"""Run missing benchmark configs in isolated subprocesses to survive OOM kills."""
import subprocess
import csv
import os
import sys

CSV_FILE = 'test/output/efficiency_budget8K_1M/decoding_time_vs_context.csv'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTOR = os.path.join(ROOT, 'L3_8Bi_d16_i512_pf4.pt')
# Prefer the interpreter running this script (handles uv/venv cases where
# CONDA_PREFIX may point to an unrelated system conda install).
PYTHON = sys.executable
if not os.path.exists(PYTHON):
    PYTHON = os.path.join(os.environ.get('CONDA_PREFIX', ''), 'bin', 'python')

# Load completed configs
completed = set()
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('status') == 'success':
                completed.add((row['label'], int(row['context_length'])))

print(f"Already completed: {len(completed)} configs")

# All configs to run
configs = [
    {'label': 'Dense', 'mode': 'full', 'mode_cpu': 'full_cpu', 'sparse_budget': 8192, 'predict_interval': 1, 'enable_neighbor_fetch': False, 'oracle_random_indices': True},
    {'label': 'KeySifter (i=1)', 'mode': 'keysifter', 'mode_cpu': 'keysifter_cpu', 'sparse_budget': 8192, 'predict_interval': 1, 'enable_neighbor_fetch': False, 'oracle_random_indices': True},
    {'label': 'KeySifter (i=2+nb)', 'mode': 'keysifter', 'mode_cpu': 'keysifter_cpu', 'sparse_budget': 8192, 'predict_interval': 2, 'enable_neighbor_fetch': True, 'oracle_random_indices': True},
    {'label': 'KeySifter (i=4+nb)', 'mode': 'keysifter', 'mode_cpu': 'keysifter_cpu', 'sparse_budget': 8192, 'predict_interval': 4, 'enable_neighbor_fetch': True, 'oracle_random_indices': True},
    {'label': 'KeySifter (i=8+nb)', 'mode': 'keysifter', 'mode_cpu': 'keysifter_cpu', 'sparse_budget': 8192, 'predict_interval': 8, 'enable_neighbor_fetch': True, 'oracle_random_indices': True},
    {'label': 'KeySifter (i=16+nb)', 'mode': 'keysifter', 'mode_cpu': 'keysifter_cpu', 'sparse_budget': 8192, 'predict_interval': 16, 'enable_neighbor_fetch': True, 'oracle_random_indices': True},
    {'label': 'Oracle (random)', 'mode': 'oracle', 'mode_cpu': 'oracle_cpu', 'sparse_budget': 8192, 'predict_interval': 1, 'enable_neighbor_fetch': False, 'oracle_random_indices': True},
    {'label': 'Oracle (contiguous)', 'mode': 'oracle', 'mode_cpu': 'oracle_cpu', 'sparse_budget': 8192, 'predict_interval': 1, 'enable_neighbor_fetch': False, 'oracle_random_indices': False},
    {'label': 'Oracle (random, i=16)', 'mode': 'oracle', 'mode_cpu': 'oracle_cpu', 'sparse_budget': 8192, 'predict_interval': 16, 'enable_neighbor_fetch': False, 'oracle_random_indices': True},
]

context_lengths = [32768, 65536, 131072, 262144, 524288, 1048576]
CPU_OFFLOAD_THRESHOLD = 131072

for ctx_len in context_lengths:
    use_cpu = ctx_len > CPU_OFFLOAD_THRESHOLD
    gen_len = 128 if use_cpu else 1024

    for cfg in configs:
        label = cfg['label']

        if (label, ctx_len) in completed:
            print(f"SKIP {label} @ {ctx_len} (already done)")
            continue

        mode = cfg['mode_cpu'] if use_cpu else cfg['mode']
        print(f"\nRUN  {label} @ {ctx_len} (mode={mode}, gen={gen_len})")

        # Run benchmark in isolated subprocess
        script = f"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'test')
from benchmark_keysifter import benchmark_model

res = benchmark_model(
    attn_mode='{mode}',
    prompt_length={ctx_len},
    gen_length={gen_len},
    sparse_budget={cfg['sparse_budget']},
    predictor_path='{PREDICTOR}',
    oracle_random_indices={cfg['oracle_random_indices']},
    predict_interval={cfg['predict_interval']},
    enable_neighbor_fetch={cfg['enable_neighbor_fetch']},
)
print(f"RESULT={{res['decode_time_avg'] * 1000.0:.4f}}")
"""
        try:
            result = subprocess.run(
                [PYTHON, '-c', script],
                capture_output=True, text=True, timeout=7200,  # 2 hour timeout per config
                cwd=ROOT,
            )

            # Parse result
            avg_ms = None
            for line in result.stdout.split('\n'):
                if line.startswith('RESULT='):
                    avg_ms = float(line.split('=')[1])

            if avg_ms is not None:
                with open(CSV_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([label, ctx_len, mode, f'{avg_ms:.4f}', 'success'])
                print(f"  OK: {avg_ms:.2f} ms/token")
            else:
                print(f"  FAIL: no result parsed")
                print(f"  stdout: {result.stdout[-500:]}")
                print(f"  stderr: {result.stderr[-500:]}")
                with open(CSV_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([label, ctx_len, mode, '0', f'error: {result.returncode}'])

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (2 hr)")
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([label, ctx_len, mode, '0', 'timeout'])
        except Exception as e:
            print(f"  ERROR: {e}")
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([label, ctx_len, mode, '0', f'error: {e}'])

print("\n=== ALL CONFIGS ATTEMPTED ===")
