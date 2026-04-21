# LLM Inference Optimization (DS-GA 1019)

Performance optimization of local LLM inference in Python. We benchmark 3 models (GPT-2 124M, TinyLlama 1.1B, Pythia 1B), profile them with `cProfile` and `line_profiler`, and apply Python optimization techniques (KV-Cache, INT8 Quantization, Async Batching, Numba, NumPy) to maximize tokens/second.

## Models

| Model | Params | Type |
|---|---|---|
| GPT-2 | 124M | Base (text completion) |
| TinyLlama-1.1B-Chat | 1.1B | Instruction-tuned |
| Pythia-1B | 1B | Base (text completion) |

## Results (CPU, 50 tokens)

| Model | Baseline | KV-Cache | Quant+KV | Batch+KV | All Combined |
|---|---|---|---|---|---|
| GPT-2 (124M) | 38 tok/s | 84 tok/s (2.2×) | 29 tok/s (0.77×) | 110 tok/s (2.9×) | 37 tok/s |
| TinyLlama (1.1B) | 4.3 tok/s | 13.4 tok/s (3.1×) | 2.6 tok/s (0.62×) | 13.0 tok/s (3.1×) | 5.1 tok/s |
| Pythia (1B) | 5.4 tok/s | 15.4 tok/s (2.9×) | 3.5 tok/s (0.64×) | 14.7 tok/s (2.7×) | 4.8 tok/s |

> KV-Cache is the best single technique (2.2–3.1× speedup). INT8 Quantization saves 43–70% memory but slows inference on CPU due to dequantize overhead.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/ranjan2601/DS-GA-1019.git
cd DS-GA-1019
uv sync
```

## Running

### Streamlit Demo App

```bash
uv run streamlit run app.py
```

Interactive demo with model selector (GPT-2, TinyLlama, Pythia), baseline vs quantized comparison, and live generation.

### Notebooks

```bash
uv run jupyter notebook
```

| Notebook | Description |
|---|---|
| `01_baseline_and_profiling.ipynb` | GPT-2 baseline, cProfile, line_profiler, memory profiling |
| `02_optimizations.ipynb` | KV-Cache, INT8 Quantization, Async Batching benchmarks |
| `03_combined_optimizations.ipynb` | All 5 optimization combinations benchmarked |
| `04_final_results.ipynb` | Final presentation — all 3 models, graphs, summary |

### Using the modules directly

```python
from src.model import load_model_and_tokenizer
from src.kv_cache import generate_with_kv_cache
from src.benchmark import run_benchmark

model, tokenizer = load_model_and_tokenizer("gpt2")
result = generate_with_kv_cache(model, tokenizer, "The meaning of life is")
print(f"{result['tok_per_sec']:.2f} tok/s")
```

## Project Structure

```
src/
├── config.py           # Models, prompts, benchmark settings
├── model.py            # Load model + tokenizer from HuggingFace
├── inference.py        # Baseline generation (manual token loop)
├── kv_cache.py         # KV-Cache optimization
├── quantization.py     # INT8 quantization (Conv1D + Linear)
├── async_batching.py   # Async batched generation
├── profiling.py        # cProfile, line_profiler, tracemalloc wrappers
└── benchmark.py        # Benchmark harness, comparison table, plots

notebooks/
├── 01_baseline_and_profiling.ipynb
├── 02_optimizations.ipynb
├── 03_combined_optimizations.ipynb
└── 04_final_results.ipynb

app.py                  # Streamlit demo app
```

## Optimization Techniques

| Technique | What it does |
|---|---|
| **KV-Cache** | Caches key/value tensors after prefill — avoids recomputing full attention on every token |
| **INT8 Quantization** | Quantizes weights to int8 — reduces model size by 43–70% |
| **Async Batching** | Batches multiple prompts into one forward pass — higher aggregate throughput |
| **Numba `@njit`** | JIT-compiled INT8 matmul kernel |
| **NumPy** | Pre-allocated buffers for KV-cache and quantization scaling |
