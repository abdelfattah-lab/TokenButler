# TokenButler: Token Importance is Predictable

This repository accompanies the paper [**TokenButler: Token Importance is Predictable**](https://arxiv.org/abs/2503.07518). It contains the code and reproduction
scripts for the long-context decoding experiments in the paper. the Llama-3.1-8B-Instruct predictor is
released on Hugging Face at
[Alwahsh/Meta-Llama-3.1-8B-Instruct-Butler](https://huggingface.co/Alwahsh/Meta-Llama-3.1-8B-Instruct-Butler)
and is fetched by `scripts/download_weights.sh` (see [Installation](#installation)).

TokenButler is a lightweight (≈1% parameter overhead) query-aware predictor that
identifies the small subset of KV-cache tokens needed by each decoding step. By
preserving the full KV cache and selecting tokens at fine granularity, TokenButler
avoids the failure modes of permanent eviction (H2O, SnapKV) and coarse paging
(Quest), and matches or surpasses prior KV-sparsity methods on long-context
benchmarks. With prediction-interval and neighbor fetching, the predictor cost is
amortized to deliver up to ≈1.6× on-GPU speedup over Dense Attention while staying
within ≈1.1% accuracy of every-step prediction, and up to ≈7.6× speedup over Dense
under CPU offloading at 1M context.

## Reproducing the paper

| Paper artifact | Script |
|---|---|
| Table 4 — RULER 64K accuracy on Llama-3.1-8B-Instruct (budget 8K, 10 tasks) | `examples/tokenbutler_accuracy.sh` |
| Table 6 — RULER 64K predict-interval ablation (budget 8K, i ∈ {1,2,4,8,16} with neighbor fetch for i>1) | `examples/tokenbutler_accuracy_intervals.sh` |
| Figure 4 — decode latency vs context (GPU 32K–128K + CPU offload 256K–1M; Dense, TokenButler intervals, Oracle) | `examples/tokenbutler_efficiency.sh` then `test/plot_decoding_efficiency.py` |
| Per-operation timing breakdown (QKV / RoPE / Attention / Predictor / Scoring / Selection / KV gather) | `test/benchmark_combined_figure.py` then `test/plot_timing_breakdown.py` |
| All of the above end-to-end | `examples/reproduce_all.sh` |

Per-stage instructions are in [Running stages individually](#running-stages-individually);
the one-shot path is in [One-shot reproduction](#one-shot-reproduction).

## Requirements

- Python 3.11
- CUDA-capable GPU. The paper used a single NVIDIA A6000 (48 GB). 64K-context
  RULER evaluation comfortably fits in 48 GB; 128K-context efficiency runs need
  similar memory; ≥256K runs use CPU offloading and require sufficient host RAM
  (≈64 GB at 1M).
- Access to `meta-llama/Meta-Llama-3.1-8B-Instruct` on Hugging Face (gated;
  `huggingface-cli login` once, then it is cached).
- For the CUDA kernels: NVCC matching your PyTorch build.

## Installation

```bash
# 1. Create the environment (uv recommended, but plain venv also works)
uv venv --python 3.11
source .venv/bin/activate
uv pip install --upgrade pip

# 2. Install Python dependencies
uv pip install -r requirements.txt
uv pip install flash-attn==2.7.4.post1 --no-build-isolation

# 3. Pull CUTLASS (required to build the CUDA kernels)
mkdir -p 3rdparty
git clone https://github.com/NVIDIA/cutlass.git 3rdparty/cutlass

# 4. Build the CUDA kernels in-place
python setup.py build_ext --inplace

# 5. Download the Llama-3.1-8B-Instruct TokenButler predictor weights
bash scripts/download_weights.sh
```

The download step fetches `L3_8Bi_d16_i512_pf4.pt` from Hugging Face
and places it at the repo root, where every script expects it. The predictor uses a producer every G=4 layers,
interaction dimension d′=16, and a two-layer MLP with hidden size 512
(matching Appendix A in the paper).

## One-shot reproduction

To run the entire reproducible portion of the paper end-to-end:

```bash
bash examples/reproduce_all.sh
```

Stages and outputs:

| # | Stage | Output |
|---|---|---|
| 1 | Build RULER benchmark data | `data/ruler/data/llama-3/{32768,65536,131072,262144}/...` |
| 2 | RULER accuracy at 64K, budget 8K | `archive/Meta-Llama-3.1-8B-Instruct/ruler/*.jsonl` |
| 3 | RULER accuracy with predict-interval i ∈ {1,2,4,8,16}, budget 8K | `archive/Meta-Llama-3.1-8B-Instruct/ruler/*.jsonl` |
| 4 | Decoding-efficiency sweep, 32K → 1M | `test/output/efficiency_budget8K_1M/decoding_time_vs_context.csv` |
| 5 | Per-operation timing breakdown | `paper_plots/combined_timing.csv`, `paper_plots/combined_timing_figure.{pdf,png}` |
| 6 | Render figures | `paper_plots/decoding_performance.pdf` (GPU), `paper_plots/decoding_performance_cpu.pdf` (CPU offload), `paper_plots/timing_breakdown.pdf` |

Each accuracy JSONL stores per-sample predictions; the final line records
`avg_score`, the metric reported in the paper. The efficiency CSV is appended
to incrementally so a re-run picks up where it left off — already-completed
`(label, context_length)` pairs are skipped.

## Running stages individually

### Build the RULER datasets

```bash
bash examples/build_ruler_data.sh
```

Generates the synthetic RULER samples (NIAH variants, VT, FWE, QA-1, QA-2) at
sequence lengths 32K, 64K, 128K, and 256K under
`data/ruler/data/llama-3/`. This is required before any accuracy run.

### RULER accuracy (Table 4)

```bash
bash examples/tokenbutler_accuracy.sh
```

Evaluates Llama-3.1-8B-Instruct + TokenButler at 64K context with sparse budget
8192 across RULER tasks (`niah_single_{1,2}`, `niah_multikey_{1,2}`,
`niah_multiquery`, `niah_multivalue`, `vt`, `fwe`, `qa_1`, `qa_2`). Results
land at
`archive/Meta-Llama-3.1-8B-Instruct/ruler/<task>_65536_TokenButler_b8192_*.jsonl`.

### Predict-interval ablation (Table 6)

```bash
bash examples/tokenbutler_accuracy_intervals.sh
```

Sweeps the prediction interval i ∈ {1, 2, 4, 8, 16} at 64K context with sparse
budget 8192. i=1 runs the predictor every step (the no-amortization baseline);
i>1 enables neighbor fetching to amortize the predictor across N steps with
minimal accuracy loss.

A finer-grained variant of this ablation, including paired efficiency
measurements and a combined accuracy/latency plot at the same 64K /
sparse-budget-8192 settings (and an extended interval i ∈ {1, 2, 4, 8, 16, 32}),
is also available:

```bash
python test/ablation_predict_interval.py            # full run, all 11 RULER tasks at 64K
python test/ablation_predict_interval.py --quick    # 15 samples, 4 datasets — sanity check
```

### Decoding efficiency (Figure 4)

```bash
bash examples/tokenbutler_efficiency.sh
```

Measures per-token decode latency on Llama-3.1-8B-Instruct at sparse budget
8192, across:

- **GPU**: contexts 32K, 64K, 128K (gen_length=1024)
- **CPU offload**: contexts 256K, 512K, 1M (gen_length=128)

For each context, it benchmarks Dense, TokenButler with i ∈ {1, 2, 4, 8, 16}
(plus neighbor fetch for i>1), and Oracle baselines (random / contiguous /
random-with-i=16) as zero-overhead lower bounds. Each configuration runs in an
isolated subprocess — see `test/run_missing_configs.py` — so OOMs or timeouts
do not poison the rest of the sweep. Results stream into
`test/output/efficiency_budget8K_1M/decoding_time_vs_context.csv`.

To plot the GPU and CPU-offload curves separately:

```bash
python test/plot_decoding_efficiency.py \
    --csv test/output/efficiency_budget8K_1M/decoding_time_vs_context.csv \
    --output-dir paper_plots
```

### Timing breakdown

```bash
python test/benchmark_combined_figure.py --predictor-path L3_8Bi_d16_i512_pf4.pt
python test/plot_timing_breakdown.py --csv paper_plots/combined_timing.csv \
    --output paper_plots/timing_breakdown.pdf
```

`benchmark_combined_figure.py` profiles every component (QKV proj, RoPE,
flash-attn, MLP, predictor forward, score computation, top-K selection, key/
value gather) at multiple K values, writes a tidy CSV, and emits a combined
figure. `plot_timing_breakdown.py` re-renders the stacked-area figure used in
the paper from that CSV.

## Standalone single-command evaluation

For quick experimentation on one dataset:

```bash
python test/eval_acc.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --method TokenButler \
    --datalen 65536 \
    --dataset_name ruler/niah_single_1 \
    --sparse_budget 8192 \
    --chunk_size 8 \
    --rank 160 \
    --predictor_path L3_8Bi_d16_i512_pf4.pt \
    --dDash 16 \
    --producer_frequency 4 \
    --tokenbutler_intermediate_dim 512
```

Useful flags for the predict-interval optimization:

- `--predict_interval N`: invoke the predictor every N decode steps (default 1).
- `--enable_neighbor_fetch`: when `predict_interval > 1`, also fetch neighbors
  of selected tokens to mitigate stale selections.

## Repository layout

```
.
├── models/
│   ├── tokenbutler_predictor.py    # the lightweight predictor (paper §3.1)
│   ├── kv_cache_tokenbutler.py     # GPU-resident sparse KV cache
│   ├── kv_cache_tokenbutler_cpu.py # CPU-offload sparse KV cache
│   ├── kv_cache_oracle{,_cpu}.py   # Oracle baselines for the efficiency study
│   ├── kv_cache_dsa.py             # DeepSeek-V3.2-style indexer baseline
│   ├── kv_cache.py                 # Dense + ShadowKV references
│   ├── llama.py / glm.py / qwen.py / phi3.py
│   └── base.py                     # LLM wrapper + cache factory
├── kernels/
│   ├── int8_score_fused.py
│   └── *.cu                        # CUTLASS-backed prefill kernels
├── data/
│   ├── dataset.py                  # unified dataset interface
│   ├── metrics.py
│   └── ruler/                      # RULER generation (adapted from RULER repo)
├── test/
│   ├── eval_acc.py                 # accuracy evaluation entry point
│   ├── evaluator.py
│   ├── benchmark_tokenbutler.py    # decode-latency micro-benchmark
│   ├── run_missing_configs.py      # efficiency-sweep orchestrator (OOM-safe)
│   ├── benchmark_combined_figure.py# per-operation timing breakdown
│   ├── plot_decoding_efficiency.py # Figure 4 renderer
│   ├── plot_timing_breakdown.py    # timing-breakdown renderer
│   ├── ablation_predict_interval.py# Table-6-style ablation with paired plot
│   └── benchmark_accuracy.py       # automated comparison driver (optional)
├── examples/
│   ├── reproduce_all.sh
│   ├── build_ruler_data.sh
│   ├── tokenbutler_accuracy.sh
│   ├── tokenbutler_accuracy_intervals.sh
│   └── tokenbutler_efficiency.sh
├── scripts/
│   └── download_weights.sh
├── 3rdparty/cutlass/               # populated by `git clone` step above
├── requirements.txt
├── setup.py                        # builds the CUTLASS-backed CUDA kernels
└── TokenButler.pdf
```

## Supported models

The released predictor (downloaded from
[`Alwahsh/Meta-Llama-3.1-8B-Instruct-Butler`](https://huggingface.co/Alwahsh/Meta-Llama-3.1-8B-Instruct-Butler))
targets **Llama-3.1-8B-Instruct** and reproduces the Llama rows of Tables 4
and 6 and the entirety of Figure 4. Architecture support exists in
`models/{glm,qwen,phi3}.py` for evaluating with predictors trained on those
backbones.

## Citation

If you use TokenButler in your work, please cite the paper:

```bibtex
@misc{akhauri2025tokenbutlertokenimportancepredictable,
      title={TokenButler: Token Importance is Predictable}, 
      author={Yash Akhauri and Ahmed F AbouElhamayed and Yifei Gao and Chi-Chih Chang and Nilesh Jain and Mohamed S. Abdelfattah},
      year={2025},
      eprint={2503.07518},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.07518}, 
}
```

## License

Apache 2.0. See [`LICENSE`](LICENSE).

## Acknowledgements

This repository builds directly on:

- **xKV** (Chang et al., 2025) — provided the starting point for this codebase.
- **RULER** (Hsieh et al., 2024) — synthetic long-context benchmark; the
  generation code under `data/ruler/` is adapted from the original repository.

The TokenButler-specific contributions in this release — the predictor module,
the GPU and CPU-offload sparse KV caches, and the prediction-interval /
neighbor-fetching scheme — are described in the accompanying paper.
