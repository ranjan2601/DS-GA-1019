"""
Benchmarking harness for controlled, reproducible inference measurements.

Usage:
    from src.benchmark import run_benchmark
    results = run_benchmark(model, tokenizer, generate_fn, label="baseline")
"""

import time
import statistics
import json
import os
from datetime import datetime

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.config import (
    BENCHMARK_PROMPTS,
    BENCHMARK_RUNS,
    MAX_NEW_TOKENS,
    WARMUP_RUNS,
)


def _run_single(model, tokenizer, generate_fn, prompt, max_new_tokens):
    """Run one generation and return the result dict."""
    return generate_fn(model, tokenizer, prompt, max_new_tokens=max_new_tokens)


def run_benchmark(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    generate_fn,
    label: str = "baseline",
    prompts: list[str] | None = None,
    num_runs: int = BENCHMARK_RUNS,
    warmup_runs: int = WARMUP_RUNS,
    max_new_tokens: int = MAX_NEW_TOKENS,
    save_path: str | None = None,
) -> dict:
    """
    Run a full benchmark suite.

    For each prompt:
      1. Discard `warmup_runs` runs (cold cache, JIT warm-up).
      2. Time `num_runs` runs, collect tokens/sec for each.

    Returns a summary dict with per-prompt and aggregate statistics.
    """
    prompts = prompts or BENCHMARK_PROMPTS

    all_tok_per_sec = []
    per_prompt_results = []

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] Benchmarking: {prompt[:50]}...")

        # warm-up
        for _ in range(warmup_runs):
            _run_single(model, tokenizer, generate_fn, prompt, max_new_tokens)

        # timed runs
        prompt_speeds = []
        for run_idx in range(num_runs):
            result = _run_single(
                model, tokenizer, generate_fn, prompt, max_new_tokens
            )
            prompt_speeds.append(result["tok_per_sec"])

        avg = statistics.mean(prompt_speeds)
        std = statistics.stdev(prompt_speeds) if len(prompt_speeds) > 1 else 0.0
        all_tok_per_sec.extend(prompt_speeds)

        per_prompt_results.append({
            "prompt": prompt,
            "runs": num_runs,
            "tok_per_sec_mean": round(avg, 2),
            "tok_per_sec_std": round(std, 2),
            "tok_per_sec_all": [round(s, 2) for s in prompt_speeds],
        })

    summary = {
        "label": label,
        "model": str(model.config._name_or_path),
        "max_new_tokens": max_new_tokens,
        "num_prompts": len(prompts),
        "runs_per_prompt": num_runs,
        "warmup_runs": warmup_runs,
        "overall_tok_per_sec_mean": round(statistics.mean(all_tok_per_sec), 2),
        "overall_tok_per_sec_std": round(
            statistics.stdev(all_tok_per_sec) if len(all_tok_per_sec) > 1 else 0.0, 2
        ),
        "per_prompt": per_prompt_results,
        "timestamp": datetime.now().isoformat(),
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Results saved to {save_path}")

    return summary


def compare_benchmarks(results: list[dict]) -> None:
    """
    Print a comparison table across multiple benchmark runs.
    """
    print(f"\n{'Label':<25} {'tok/s (mean)':>14} {'tok/s (std)':>14} {'Speedup':>10}")
    print("-" * 65)

    baseline_speed = results[0]["overall_tok_per_sec_mean"] if results else 1.0

    for r in results:
        speed = r["overall_tok_per_sec_mean"]
        std = r["overall_tok_per_sec_std"]
        speedup = speed / baseline_speed if baseline_speed > 0 else 0.0
        print(f"{r['label']:<25} {speed:>14.2f} {std:>14.2f} {speedup:>9.2f}x")


def plot_speedups(results: list[dict], save_path: str | None = None):
    """
    Bar chart of tok/sec across optimization stages.
    """
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in results]
    means = [r["overall_tok_per_sec_mean"] for r in results]
    stds = [r["overall_tok_per_sec_std"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color="steelblue", edgecolor="black")

    ax.set_ylabel("Tokens / Second")
    ax.set_title("Inference Throughput by Optimization Stage")
    ax.set_ylim(bottom=0)

    # annotate bars
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Plot saved to {save_path}")

    return fig
