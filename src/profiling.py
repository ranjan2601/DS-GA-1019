"""
Profiling utilities for identifying inference bottlenecks.

Provides wrappers for:
- cProfile (function-level)
- line_profiler (line-level hotspots)
- memory_profiler (memory usage over time)
"""

import cProfile
import pstats
import io
import os
import tracemalloc

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


# ── line_profiler ────────────────────────────────────────────────────────────


def _unwrap(func):
    """Unwrap decorated functions (e.g., @torch.no_grad) to get the real function."""
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func


def line_profile_function(func, *args, **kwargs):
    """
    Run line_profiler on a single function call and return the report string.

    Automatically unwraps decorators like @torch.no_grad() so that the
    actual function body is profiled, not the wrapper.

    Usage:
        from src.inference import generate_manual
        report = line_profile_function(generate_manual, model, tokenizer, "Hello")
    """
    from line_profiler import LineProfiler

    inner_func = _unwrap(func)
    lp = LineProfiler()
    lp.add_function(inner_func)
    wrapped = lp(func)
    result = wrapped(*args, **kwargs)

    stream = io.StringIO()
    lp.print_stats(stream=stream)

    return {
        "result": result,
        "report": stream.getvalue(),
    }


def line_profile_with_subroutines(func, sub_functions, *args, **kwargs):
    """
    Run line_profiler on a function AND specified sub-functions.

    This lets you profile the generate loop AND the inner functions
    it calls (e.g., model forward, softmax, sampling).

    Usage:
        report = line_profile_with_subroutines(
            generate_manual,
            [torch.softmax, torch.multinomial],
            model, tokenizer, "Hello",
        )
    """
    from line_profiler import LineProfiler

    inner_func = _unwrap(func)
    lp = LineProfiler()
    lp.add_function(inner_func)
    for sub_fn in sub_functions:
        lp.add_function(sub_fn)

    wrapped = lp(func)
    result = wrapped(*args, **kwargs)

    stream = io.StringIO()
    lp.print_stats(stream=stream)

    return {
        "result": result,
        "report": stream.getvalue(),
    }


# ── memory_profiler ──────────────────────────────────────────────────────────


def memory_profile_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    generate_fn,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    """
    Track Python memory allocation during a generation call using tracemalloc.

    Returns peak memory, current memory, and top allocation sites.
    """
    tracemalloc.start()

    result = generate_fn(model, tokenizer, prompt, max_new_tokens=max_new_tokens)

    snapshot = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Top 15 allocation sites
    top_stats = snapshot.statistics("lineno")
    top_allocations = []
    for stat in top_stats[:15]:
        top_allocations.append({
            "file": str(stat.traceback),
            "size_kb": stat.size / 1024,
            "count": stat.count,
        })

    return {
        "generation_result": result,
        "current_memory_mb": current / (1024 ** 2),
        "peak_memory_mb": peak / (1024 ** 2),
        "top_allocations": top_allocations,
    }
