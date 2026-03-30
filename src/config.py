"""
Central configuration for the LLM inference optimization project.
"""

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = "gpt2"  # HuggingFace model id (GPT-2 124M)
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
