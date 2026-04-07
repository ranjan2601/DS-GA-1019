# LLM Inference Optimization (DS-GA 1019)

Performance optimization of local LLM inference in Python. We take GPT-2 124M, profile it, and apply Python optimization techniques (Cython, Numba, NumPy, asyncio) to maximize tokens/second — measuring speedups at every step.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Clone the repo
git clone https://github.com/ranjan2601/DS-GA-1019.git
cd DS-GA-1019

# Install dependencies
uv sync
```

This creates a `.venv/` with all pinned dependencies.

## Running

### Jupyter Notebook

```bash
uv run jupyter notebook notebooks/01_baseline_and_profiling.ipynb
```

This runs the full Phase 1 pipeline:
1. Loads GPT-2 124M
2. Measures baseline tokens/sec
3. Profiles with cProfile to identify bottlenecks
4. Runs the benchmark suite (10 prompts × 5 runs × 200 tokens)
5. Quality sanity check on factual prompts
6. Generates comparison plots

### Using the modules directly

```python
from src.model import load_model_and_tokenizer
from src.inference import generate, generate_manual
from src.benchmark import run_benchmark

model, tokenizer = load_model_and_tokenizer()

# Single generation
result = generate(model, tokenizer, "The meaning of life is")
print(f"{result['tok_per_sec']:.2f} tok/s")

# Full benchmark
results = run_benchmark(model, tokenizer, generate, label="baseline")
```

## Project Structure

```
src/
├── config.py       # Model name, prompts, benchmark settings
├── model.py        # Load model + tokenizer from HuggingFace
├── inference.py    # Baseline generation (HF built-in + manual loop)
├── profiling.py    # cProfile wrapper + memory footprint
└── benchmark.py    # Benchmark harness, comparison table, plots

notebooks/
└── 01_baseline_and_profiling.ipynb  # Phase 1 demo notebook
```
