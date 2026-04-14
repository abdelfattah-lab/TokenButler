# xKV: Efficient KV Cache Management with Learned Token Importance

xKV introduces TokenButler, a method that uses a lightweight neural predictor to identify important tokens in the KV cache during LLM decoding. By selectively attending to only the most relevant tokens, TokenButler achieves significant speedups over dense attention while maintaining accuracy on long-context tasks. It supports both GPU-only inference (up to 128K context) and CPU-offloading mode (up to 1M+ context).

## Installation

### Requirements

- Python 3.11+
- CUDA-capable GPU (A100 80GB recommended for long contexts)
- ~20GB GPU memory for 128K context evaluation

### Setup

```bash
# Create environment
uv venv --python 3.11 && source .venv/bin/activate && uv pip install --upgrade pip

# Install packages
uv pip install -r requirements.txt
uv pip install flash-attn==2.7.4.post1 --no-build-isolation

# Clone CUTLASS (required for CUDA kernel compilation)
mkdir -p 3rdparty
git clone https://github.com/NVIDIA/cutlass.git 3rdparty/cutlass

# Build CUDA kernels
python setup.py build_ext --inplace

# Download TokenButler predictor weights
bash scripts/download_weights.sh
```

## Quick Start

```bash
# 1. Build RULER benchmark data
bash examples/build_ruler_data.sh

# 2. Run accuracy evaluation on a single dataset
python test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --method KeySifter \
    --datalen 65536 \
    --dataset_name ruler/niah_single_1 \
    --sparse_budget 2048 \
    --chunk_size 8 \
    --rank 160 \
    --predictor_path L3_8Bi_d16_i512_pf4.pt \
    --dDash 16 \
    --producer_frequency 4 \
    --keysifter_intermediate_dim 512
```

## Reproducing Paper Results

### Accuracy on RULER Benchmark

Evaluates TokenButler on 10 RULER datasets (N-S1, N-S2, N-MK1, N-MK2, N-MQ, N-MV, QA-1, QA-2, VT, FWE) with Llama-3.1-8B-Instruct at 65K context length:

```bash
bash examples/tokenbutler_accuracy.sh
```

Results are saved as JSONL files in `archive/Meta-Llama-3.1-8B-Instruct/`. The final line of each file contains the `avg_score` (accuracy).

### Accuracy with Prediction Intervals

Tests prediction step intervals (1, 2, 4, 8, 16) with neighbor fetching, showing the accuracy/efficiency tradeoff:

```bash
bash examples/tokenbutler_accuracy_intervals.sh
```

- **Interval 1**: Prediction at every decode step (highest accuracy, highest cost)
- **Interval N + neighbor fetch**: Predict every N steps, fetch neighboring tokens to compensate (more efficient)

### Decoding Efficiency

Measures decode latency (ms/token) with 8K sparse token budget across context lengths:

- **GPU** (32K, 64K, 128K): generation length 1024 tokens
- **CPU offloading** (256K, 512K, 1M): generation length 128 tokens

```bash
bash examples/tokenbutler_efficiency.sh
```

Results are saved to `test/output/efficiency_budget8K_1M/decoding_time_vs_context.csv`.

For the accuracy benchmarks with automated comparison tables, you can also use:

```bash
python test/benchmark_accuracy.py              # full evaluation
python test/benchmark_accuracy.py --quick      # quick test (15 samples/dataset)
```

## Repository Structure

```
xKV/
├── models/                     # Model implementations and KV cache variants
│   ├── base.py                 # Base LLM class with cache factory
│   ├── llama.py                # Llama model wrapper
│   ├── keysifter_predictor.py  # TokenButler importance predictor
│   ├── kv_cache_keysifter.py   # TokenButler sparse cache (GPU)
│   ├── kv_cache_keysifter_cpu.py  # TokenButler with CPU offloading
│   ├── kv_cache.py             # Dense and ShadowKV caches
│   ├── kv_cache_cpu.py         # Dense attention with CPU offloading
│   └── tensor_op.py            # CUDA tensor operations
├── kernels/                    # CUDA and Triton kernels
│   ├── int8_score_fused.py     # INT8 importance scoring (Triton)
│   ├── keysifter_score_triton.py  # Importance scoring kernel
│   └── *.cu                    # CUDA kernel sources
├── data/                       # Dataset and evaluation utilities
│   ├── dataset.py              # Unified dataset interface
│   ├── metrics.py              # Evaluation metrics
│   └── ruler/                  # RULER benchmark data generation
├── test/                       # Evaluation and benchmarking scripts
│   ├── eval_acc.py             # Main accuracy evaluation
│   ├── evaluator.py            # Evaluation loop
│   ├── benchmark_keysifter.py  # Efficiency benchmarking
│   ├── benchmark_accuracy.py   # Automated accuracy comparison
│   └── run_missing_configs.py  # Efficiency benchmark orchestrator
├── examples/                   # Convenience scripts to reproduce results
├── scripts/                    # Setup utilities
└── requirements.txt
```

## Supported Models

- **Llama-3.1-8B-Instruct** (fully tested, paper results)
- GLM, Qwen2, Phi3 (architecture support, experimental)

## Citation

```bibtex
@article{xkv2025,
    title={xKV: Efficient KV Cache Management with Learned Token Importance},
    author={TODO},
    year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgements

Built upon [ShadowKV](https://github.com/bytedance/ShadowKV). RULER benchmark adapted from [RULER](https://github.com/hsiehjackson/RULER).
