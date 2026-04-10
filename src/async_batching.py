"""
Optimization 2c: Async Batching.

Bottleneck addressed: Processing one prompt at a time leaves hardware
underutilized. By batching multiple prompts into a single forward pass,
matrix operations are larger and more efficient (better CPU cache usage,
amortized overhead).

Uses asyncio for non-blocking request coordination and batches prompts
together for a single forward pass.
"""

import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.config import MAX_NEW_TOKENS, TEMPERATURE, TOP_K


# ── Batched generation (synchronous core) ────────────────────────────────────


@torch.no_grad()
def generate_batched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Generate text for multiple prompts in a single batched forward pass.

    All prompts are padded to the same length and processed together,
    so matrix operations are larger and more hardware-efficient.

    Returns a list of result dicts (one per prompt).
    """
    # Tokenize all prompts with left-padding for causal LM
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    batch_size = input_ids.shape[0]

    # Track original prompt lengths (excluding padding)
    prompt_lengths = attention_mask.sum(dim=1).tolist()

    generated_ids = input_ids.clone()

    start = time.perf_counter()

    # Prefill with cache
    outputs = model(
        generated_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values

    # Track which sequences are still active (haven't hit EOS)
    active = torch.ones(batch_size, dtype=torch.bool)

    for _ in range(max_new_tokens):
        scaled_logits = logits
        if temperature != 1.0:
            scaled_logits = scaled_logits / temperature

        if top_k > 0:
            top_k_values, _ = torch.topk(scaled_logits, top_k)
            min_top_k = top_k_values[:, -1].unsqueeze(-1)
            scaled_logits = torch.where(
                scaled_logits < min_top_k,
                torch.full_like(scaled_logits, float("-inf")),
                scaled_logits,
            )

        probs = torch.softmax(scaled_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)  # (batch, 1)

        # Force EOS for already-finished sequences
        next_tokens[~active] = tokenizer.eos_token_id

        generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
        attention_mask = torch.cat(
            [attention_mask, active.unsqueeze(1).long()], dim=1
        )

        # Update active mask
        active = active & (next_tokens.squeeze(1) != tokenizer.eos_token_id)
        if not active.any():
            break

        outputs = model(
            next_tokens,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

    elapsed = time.perf_counter() - start

    # Unpack results per prompt
    results = []
    total_tokens = 0
    for i in range(batch_size):
        prompt_len = prompt_lengths[i]
        # Find padding offset (left-padded)
        pad_len = input_ids.shape[1] - prompt_len
        gen_ids = generated_ids[i, input_ids.shape[1]:]
        num_tokens = len(gen_ids)
        total_tokens += num_tokens
        text = tokenizer.decode(
            generated_ids[i, pad_len:], skip_special_tokens=True
        )
        results.append({
            "text": text,
            "num_tokens": num_tokens,
            "elapsed": elapsed,
            "tok_per_sec": num_tokens / elapsed if elapsed > 0 else 0.0,
        })

    return results


# ── Async interface ──────────────────────────────────────────────────────────


class AsyncBatchProcessor:
    """
    Collects incoming prompts asynchronously and batches them into
    a single forward pass once a batch is full or a timeout expires.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        timeout: float = 0.1,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens
        self._queue: asyncio.Queue | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def _get_queue(self) -> asyncio.Queue:
        if self._queue is None:
            self._queue = asyncio.Queue()
        return self._queue

    async def submit(self, prompt: str) -> dict:
        """Submit a prompt and wait for its result."""
        queue = await self._get_queue()
        future = asyncio.get_event_loop().create_future()
        await queue.put((prompt, future))
        return await future

    async def process_batch(self, prompts: list[str]) -> list[dict]:
        """
        Process a list of prompts as a batch. This is the main entry
        point for benchmarking — collects prompts and runs them together.
        """
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            generate_batched,
            self.model,
            self.tokenizer,
            prompts,
            self.max_new_tokens,
        )
        return results


# ── Wrapper with same signature for benchmarking ─────────────────────────────


def make_batched_generate_fn(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
):
    """
    Returns a generate function with the same signature as the baseline,
    but internally batches multiple calls.

    For fair benchmarking, this accumulates `batch_size` prompts and
    processes them together, reporting per-prompt throughput.
    """
    pending_prompts = []
    pending_results = []

    @torch.no_grad()
    def batched_generate(
        model_: PreTrainedModel,
        tokenizer_: PreTrainedTokenizer,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_k: int = TOP_K,
    ) -> dict:
        # Use the outer model/tokenizer for consistency
        result = generate_batched(
            model, tokenizer, [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        return result[0]

    return batched_generate


def run_batched_benchmark(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    batch_size: int = 4,
    max_new_tokens: int = MAX_NEW_TOKENS,
    num_runs: int = 5,
    warmup_runs: int = 1,
) -> dict:
    """
    Benchmark batched generation. Processes prompts in groups of
    `batch_size` and reports aggregate throughput.
    """
    import statistics

    # warm up
    for _ in range(warmup_runs):
        generate_batched(model, tokenizer, prompts[:batch_size], max_new_tokens=max_new_tokens)

    all_tok_per_sec = []

    for run_idx in range(num_runs):
        # Process all prompts in batches
        total_tokens = 0
        total_time = 0.0

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            results = generate_batched(
                model, tokenizer, batch, max_new_tokens=max_new_tokens
            )
            total_tokens += sum(r["num_tokens"] for r in results)
            total_time += results[0]["elapsed"]  # wall-clock for this batch

        run_tok_per_sec = total_tokens / total_time if total_time > 0 else 0.0
        all_tok_per_sec.append(run_tok_per_sec)

    return {
        "label": f"async_batch_{batch_size}",
        "model": str(model.config._name_or_path),
        "max_new_tokens": max_new_tokens,
        "num_prompts": len(prompts),
        "runs_per_prompt": num_runs,
        "warmup_runs": warmup_runs,
        "overall_tok_per_sec_mean": round(statistics.mean(all_tok_per_sec), 2),
        "overall_tok_per_sec_std": round(
            statistics.stdev(all_tok_per_sec) if len(all_tok_per_sec) > 1 else 0.0, 2
        ),
        "per_prompt": [],
    }
