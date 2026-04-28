# LLM Inference Optimization (DS-GA 1019)

**NYU Tandon School of Engineering — Advanced Python for Data Science**
Authors: Ranjan Patil (sp8171), Samridh Srivastava (ss18906)

Performance optimization of local LLM inference in Python. We benchmark 3 models (GPT-2 124M, TinyLlama 1.1B, Pythia 1B), profile them with `cProfile`, `line_profiler`, and `tracemalloc`, and apply Python optimization techniques (KV-Cache, INT8 Quantization with Numba, Async Batching) to maximize tokens/second on CPU.

---

## Quick Start (3 steps)

### 1. Install `uv` (Python package manager)

If you do not have `uv` installed, run one of the following:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or use pip: `pip install uv`. Full install instructions: [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/)

### 2. Clone the repo and install dependencies

```bash
git clone https://github.com/ranjan2601/DS-GA-1019.git
cd DS-GA-1019
uv sync
```

`uv sync` creates a `.venv/` directory and installs all pinned dependencies from `pyproject.toml` and `uv.lock` in under 30 seconds. This guarantees the exact versions used in our benchmarks.

### 3. Run the final results notebook

```bash
uv run jupyter notebook notebooks/04_final_results.ipynb
```

This notebook is the single best entry point. It loads cached benchmark results from `benchmark_results/` (no re-benchmarking needed) and regenerates all plots and summary tables in under 30 seconds. If you want to re-run the full benchmarks from scratch, delete the JSON files in `benchmark_results/` first (takes roughly 30 minutes on Apple M3).

---

## Running the Streamlit Demo

Interactive side-by-side comparison of baseline vs. quantized generation across all 3 models:

```bash
uv run streamlit run app.py
```

Open the URL printed in the terminal (typically `http://localhost:8501`). Features:

- Dropdown to select any of the 3 models
- Live generation with real-time tokens/sec measurement
- Baseline vs. INT8 quantized side-by-side
- Automatic chat-template handling for TinyLlama (instruction-tuned)
- Memory footprint comparison

---

## Benchmark Results (per notebook)

### Notebook 01: Baseline and Profiling (GPT-2, 10 prompts, 5 runs, 200 tokens)

Compares HuggingFace's built-in `.generate()` (uses KV-Cache internally) vs. our manual autoregressive loop (no cache). The gap is what we aim to close with our own KV-Cache implementation.

| Method | tok/s | Std |
|---|---|---|
| HuggingFace `generate()` (has internal KV-Cache) | 84.73 | +/- 3.38 |
| Manual loop (baseline, no cache) | 18.89 | +/- 2.88 |

Profiling output (this notebook): 96.7% of time on line `outputs = model(generated_ids)`, 72% cumulative in model forward pass, 51% in `torch.addmm`. FP32 model memory: 474.7 MB.

### Notebook 02: Individual Optimizations (GPT-2, 10 prompts, 5 runs, 200 tokens)

Each optimization benchmarked independently against the manual baseline.

| Config | tok/s | Std | Speedup |
|---|---|---|---|
| Baseline | 18.34 | +/- 5.58 | 1.00x |
| KV-Cache | 72.95 | +/- 2.00 | 3.98x |
| INT8 Quantized (no KV) | 23.94 | +/- 0.45 | 1.31x |

Memory: INT8 quantization reduced GPT-2 from 474.7 MB to 268.5 MB (43.4% reduction).

### Notebook 03: Combined Optimizations (GPT-2 Deep Dive, 10 prompts, 5 runs, 200 tokens)

| Config | tok/s | Speedup | Memory |
|---|---|---|---|
| Baseline | 17.92 | 1.00x | 474.7 MB |
| KV-Cache | 74.65 | 4.17x | 474.7 MB |
| Quant + KV | 22.44 | 1.25x | 268.5 MB |
| **Batch + KV (bs=4)** | **130.76** | **7.30x** | 474.7 MB |
| All Combined | 47.49 | 2.65x | 268.5 MB |

### Notebook 04: Cross-Model Comparison (5 prompts, 5 runs, 50 new tokens, batch size = 2 for batched configs, identical settings across all 3 models)

| Model | Baseline | KV-Cache | Quant + KV | Batch + KV | All Combined |
|---|---|---|---|---|---|
| GPT-2 (124M) | 38.1 tok/s | 83.7 tok/s (2.20x) | 29.2 tok/s (0.77x) | **110.5 tok/s (2.90x)** | 37.1 tok/s (0.97x) |
| TinyLlama (1.1B) | 4.25 tok/s | **13.4 tok/s (3.15x)** | 2.63 tok/s (0.62x) | 13.0 tok/s (3.07x) | 5.06 tok/s (1.19x) |
| Pythia (1B) | 5.38 tok/s | **15.4 tok/s (2.87x)** | 3.46 tok/s (0.64x) | 14.7 tok/s (2.74x) | 4.79 tok/s (0.89x) |

### Memory Reduction (INT8 Quantization)

| Model | FP32 | INT8 | Reduction |
|---|---|---|---|
| GPT-2 (124M) | 474.7 MB | 268.5 MB | 43.4% |
| TinyLlama (1.1B) | 4196.4 MB | 1236.9 MB | 70.5% |
| Pythia (1B) | 3859.6 MB | 1260.9 MB | 67.3% |

### Key Findings

- **KV-Cache is the universal win:** 2.2x-4.2x speedup across all models, zero memory cost, zero quality loss.
- **INT8 Quantization saves memory but slows CPU inference:** 43-70% memory reduction, but 0.62x-0.77x speedup. The dequantize-on-every-forward-pass overhead outweighs memory-bandwidth savings at this model scale. On INT8-native hardware (Intel VNNI, NVIDIA Tensor Cores), this would flip to a speedup.
- **Combinations do not compound:** "All Combined" (2.65x on GPT-2) underperforms Batch+KV (7.30x). Dequantization overhead cancels the batching gain.
- **96.7% of time is in the forward pass** (confirmed via `line_profiler`). Optimizing tokenization or sampling would have been a dead end.

### Benchmark Environment and Run-to-Run Variance

All measurements taken on: Apple M3 CPU only (no GPU/MPS), Python 3.11, PyTorch 2.11, FP32. Apple Silicon has unified memory and wider SIMD than typical x86 laptops, so absolute throughput will differ on other hardware.

**Expected variance even on the same machine:** roughly **+/- 30%** in absolute tok/s between independent runs of the same notebook. This comes from:

- Background OS processes competing for CPU
- Thermal state of the laptop (warm vs. cold start)
- HuggingFace tokenizer/model loading caching effects
- Python multiprocessing overhead variance
- Numba JIT compile-cache state on first call

For example, two runs of Notebook 03 produced these results:

| Config | Run 1 (200 tokens) | Run 2 (200 tokens) | Delta |
|---|---|---|---|
| Baseline | 17.92 tok/s | 20.29 tok/s | +13% |
| KV-Cache | 74.65 tok/s | 70.76 tok/s | -5% |
| Quant + KV | 22.44 tok/s | 24.47 tok/s | +9% |
| Batch + KV | 130.76 tok/s | 173.46 tok/s | +33% |
| All Combined | 47.49 tok/s | 60.26 tok/s | +27% |

Despite absolute differences, **relative speedups remain stable**: KV-Cache is always 3.5-4.2x, Batch+KV is always 7-9x peak, and Quant+KV is always 1.2-1.5x. The qualitative ordering of optimizations does not change.

On different hardware (x86 laptops, server CPUs, GPUs), absolute numbers can shift by another +/- 30-50% on top of run-to-run variance. See `report.tex` section "Reproducibility and Hardware Sensitivity" for full discussion.

---

## Repository Structure

```
DS-GA-1019/
|-- src/                           # Core optimization modules
|   |-- config.py                  # Models (3), prompts, benchmark settings, chat templates
|   |-- model.py                   # HuggingFace model + tokenizer loader
|   |-- inference.py               # Baseline: manual autoregressive token loop
|   |-- kv_cache.py                # Optimization 1: KV-Cache with past_key_values
|   |-- quantization.py            # Optimization 2: INT8 weights + Numba @njit matmul kernel
|   |-- async_batching.py          # Optimization 3: asyncio + batched forward pass
|   |-- profiling.py               # cProfile, line_profiler, tracemalloc wrappers
|   `-- benchmark.py               # Timing harness, comparison table, plots
|
|-- notebooks/                     # Jupyter notebooks (run in order)
|   |-- 01_baseline_and_profiling.ipynb   # GPT-2 baseline + 3-tool profiling
|   |-- 02_optimizations.ipynb            # Each optimization individually + benchmarks
|   |-- 03_combined_optimizations.ipynb   # All 5 configurations on GPT-2
|   `-- 04_final_results.ipynb            # All 3 models, graphs, final summary
|
|-- benchmark_results/             # Saved JSON results + comparison plots
|   |-- <model>_baseline.json      # Per-run raw tok/s measurements
|   |-- <model>_kv_cache.json
|   |-- <model>_quant_kv.json
|   |-- <model>_batch_kv.json
|   |-- <model>_all_combined.json
|   `-- *.png                      # Generated comparison charts
|
|-- app.py                         # Streamlit interactive demo
|-- report.tex                     # IEEE-format 4-page final report
|-- pyproject.toml                 # Dependencies (pinned via uv.lock)
|-- uv.lock                        # Locked versions for reproducibility
`-- README.md                      # This file
```

### What each Python module does

| File | Purpose |
|---|---|
| `src/config.py` | Central config: `AVAILABLE_MODELS` dict, benchmark prompts, chat-template helper for instruction-tuned models |
| `src/model.py` | `load_model_and_tokenizer(model_id)` - accepts any HuggingFace model id, sets eval mode, fixes pad token |
| `src/inference.py` | `generate_manual()` - baseline token-by-token loop; wrapped with `@torch.no_grad()` for profiling |
| `src/kv_cache.py` | `generate_with_kv_cache()` - prefills prompt once, caches K/V, feeds only new tokens per step |
| `src/quantization.py` | `QuantizedLinear` module (handles both `nn.Linear` and GPT-2's `Conv1D`), `quantize_model()`, `generate_quantized()`, Numba-compiled INT8 matmul kernel |
| `src/async_batching.py` | `AsyncBatchProcessor` (asyncio queue), `generate_batched()` (left-padded batch forward pass), `run_batched_benchmark()` |
| `src/profiling.py` | `cprofile_function()`, `line_profile_function()` (unwraps `@torch.no_grad` decorator), `memory_profile_generate()` (tracemalloc) |
| `src/benchmark.py` | `run_benchmark()` (warmup + N timed runs, mean/std), `compare_benchmarks()`, `plot_speedups()` |

### What each notebook does

| Notebook | Contents |
|---|---|
| `01_baseline_and_profiling.ipynb` | Loads GPT-2, runs manual generation loop, profiles with cProfile + line_profiler + tracemalloc, identifies 96.7% bottleneck in forward pass |
| `02_optimizations.ipynb` | Implements each of the 3 optimizations individually; demonstrates KV-Cache correctness (output matches baseline), memory reduction from quantization, batching speedup |
| `03_combined_optimizations.ipynb` | Benchmarks all 5 configurations on GPT-2 (baseline, KV-Cache, Quant+KV, Batch+KV, All Combined) at the deep-dive setting (200 tokens, 10 prompts, 5 runs). Produces the 7.30x peak speedup result |
| `04_final_results.ipynb` | Cross-model comparison: same 5 configurations across GPT-2, TinyLlama, Pythia at identical settings (50 tokens, 5 prompts, 5 runs). Loads from `benchmark_results/` JSON cache if available; regenerates all summary graphs |

---

## Optimization Techniques (Summary)

| Technique | What it does | Result |
|---|---|---|
| **KV-Cache** | Caches key/value tensors after prefill; each new token runs O(1) instead of O(n) | 2.2x-4.2x speedup, zero quality loss |
| **INT8 Quantization** | Replaces `nn.Linear` and `Conv1D` weights with INT8 + FP32 scale; Numba-compiled matmul kernel | 43-70% memory reduction; slower on CPU without INT8 hardware |
| **Async Batching** | `asyncio.Queue` collects prompts; dispatches in single batched forward pass with left-padding | Up to 7.30x throughput on GPT-2 when combined with KV-Cache |

## Python Tools from DS-GA 1019 Applied

| Tool | Where |
|---|---|
| `cProfile` | `src/profiling.py`, Notebook 01 |
| `line_profiler` | `src/profiling.py` (handles `@torch.no_grad` unwrap), Notebook 01 |
| `tracemalloc` | `src/profiling.py::memory_profile_generate()`, Notebook 01 |
| NumPy | Quantization scale tensors, INT8 weight buffers, array indexing in async batching |
| Numba `@njit(parallel=True)` | `src/quantization.py::int8_matmul` kernel with `prange` |
| `asyncio` | `src/async_batching.py::AsyncBatchProcessor` |
| `matplotlib` | `src/benchmark.py::plot_speedups`, all notebook visualizations |
| Streamlit | `app.py` interactive demonstrator |

---

## Troubleshooting

- **`uv: command not found`** after install: restart your shell or run `source ~/.bashrc` (Linux) / `source ~/.zshrc` (macOS).
- **HuggingFace download rate-limited:** set `HF_TOKEN` environment variable with a free token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
- **Notebook kernel not found:** run `uv run python -m ipykernel install --user --name llm-opt` then select the `llm-opt` kernel in Jupyter.
- **TinyLlama/Pythia run slowly on your machine:** these are ~9x larger than GPT-2. The cross-model benchmark uses 50 tokens + 5 prompts to keep total runtime under 30 minutes. Reduce further in `notebooks/04_final_results.ipynb` cell 1 (`BENCH_SETTINGS`) if needed.
- **Running from scratch takes too long:** the final notebook reads cached results from `benchmark_results/*.json`. Keep those files and re-run only the plot cells.

---

## Citation

If you find this useful, the code is at [github.com/ranjan2601/DS-GA-1019](https://github.com/ranjan2601/DS-GA-1019) under the `main` branch.
