"""
Profiling utilities for identifying inference bottlenecks.

Wraps cProfile and provides helpers for line_profiler integration.
"""

import cProfile
import pstats
import io
import os

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.config import MAX_NEW_TOKENS


def profile_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    generate_fn,
    max_new_tokens: int = MAX_NEW_TOKENS,
    sort_by: str = "cumulative",
    top_n: int = 30,
) -> str:
    """
    Run cProfile on a generation call and return a formatted stats string.
    """
    pr = cProfile.Profile()
    pr.enable()
    result = generate_fn(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    pr.disable()

    stream = io.StringIO()
    stats = pstats.Stats(pr, stream=stream)
    stats.strip_dirs().sort_stats(sort_by).print_stats(top_n)

    return {
        "generation_result": result,
        "profile_stats": stream.getvalue(),
    }


def save_profile(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    generate_fn,
    output_path: str = "profiling_output/baseline.prof",
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    """
    Run cProfile and save the binary .prof file for use with
    snakeviz or pstats later.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pr = cProfile.Profile()
    pr.enable()
    result = generate_fn(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    pr.disable()
    pr.dump_stats(output_path)

    return result


def get_model_memory_footprint(model: PreTrainedModel) -> dict:
    """
    Report model memory usage.
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_mb = (param_bytes + buffer_bytes) / (1024 ** 2)

    return {
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "param_memory_mb": param_bytes / (1024 ** 2),
        "buffer_memory_mb": buffer_bytes / (1024 ** 2),
        "total_memory_mb": total_mb,
        "dtype": str(next(model.parameters()).dtype),
    }
