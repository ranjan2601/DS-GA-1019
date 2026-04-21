"""
Central configuration for the LLM inference optimization project.
"""

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = "gpt2"  # default model

AVAILABLE_MODELS = {
    "GPT-2 (124M)":       "gpt2",
    "TinyLlama (1.1B)":   "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Pythia (1B)":        "EleutherAI/pythia-1b",
}

# Models that are instruction-tuned and need a chat prompt template
CHAT_MODELS = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

def format_prompt(user_input: str, model_id: str) -> str:
    """Wrap user input in the appropriate prompt template for the model."""
    if model_id in CHAT_MODELS:
        return (
            f"<|system|>\nYou are a helpful assistant.</s>\n"
            f"<|user|>\n{user_input}</s>\n"
            f"<|assistant|>\n"
        )
    return user_input

# ── Optimization modes ────────────────────────────────────────────────────────
OPTIMIZATION_MODES = {
    "Baseline":             "baseline",
    "KV-Cache":             "kv_cache",
    "Quantized + KV-Cache": "quantized",
    "Batched (bs=4) + KV":  "batched",
}

# ── Pre-computed benchmark results (GPT-2 124M, MacBook Air M3 8-core CPU) ───
BENCHMARK_RESULTS = [
    {"label": "Baseline",          "tok_s": 17.92,  "speedup": 1.00, "mem_mb": 474.7},
    {"label": "KV-Cache",          "tok_s": 74.65,  "speedup": 4.17, "mem_mb": 474.7},
    {"label": "Quant + KV-Cache",  "tok_s": 22.44,  "speedup": 1.25, "mem_mb": 268.5},
    {"label": "Batch (bs=4) + KV", "tok_s": 130.76, "speedup": 7.30, "mem_mb": 474.7},
    {"label": "All Combined",      "tok_s": 47.49,  "speedup": 2.65, "mem_mb": 268.5},
]

BENCHMARK_RESULTS_CONTEXT = "GPT-2 124M · 10 prompts · 5 runs each · MacBook Air M3 8-core CPU · pre-computed"

# ── Baseline probe (used by the dashboard to measure live baseline) ───────────
BASELINE_PROBE_PROMPT = "The meaning of life is"
BASELINE_PROBE_TOKENS = 50

MAX_NEW_TOKENS = 200  # fixed generation length for all benchmarks
TEMPERATURE = 1.0
TOP_K = 50

# ── Benchmarking ─────────────────────────────────────────────────────────────
WARMUP_RUNS = 1
BENCHMARK_RUNS = 5

# ── Prompts ──────────────────────────────────────────────────────────────────
BENCHMARK_PROMPTS = [
    "The meaning of life is",
    "In a distant galaxy, a lone astronaut discovered",
    "The Python programming language was created by",
    "Once upon a time in a land far away,",
    "The capital of France is",
    "Explain how a neural network works:",
    "The quick brown fox jumps over the",
    "In the year 2050, technology has advanced to the point where",
    "What is 2 + 2? The answer is",
    "The three laws of thermodynamics state that",
]

# ── Quality check prompts (factual, verifiable) ─────────────────────────────
QUALITY_CHECK_PROMPTS = [
    "What is 2 + 2? The answer is",
    "The capital of France is",
    "Water freezes at",
    "The Earth orbits the",
]
