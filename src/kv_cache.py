"""
Optimization 2a: KV-Cache for autoregressive generation.

Bottleneck addressed: Without caching, the model recomputes key/value
projections for ALL previous tokens at every generation step. With the
KV-cache, we only feed the newest token and reuse cached K/V tensors,
eliminating O(seq_len) redundant computation per step.
"""

import time

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.config import MAX_NEW_TOKENS, TEMPERATURE, TOP_K


@torch.no_grad()
def generate_with_kv_cache(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
) -> dict:
    """
    Autoregressive generation with KV-cache.

    Instead of feeding the entire sequence at each step, we:
    1. Run the full prompt through the model once (prefill).
    2. Cache the key/value tensors from every attention layer.
    3. For each subsequent token, only feed the new token + cached K/V.

    This reduces per-step computation from O(seq_len) to O(1) for the
    attention key/value projections.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    start = time.perf_counter()

    # ── Prefill: process the full prompt, get initial cache ──────────
    outputs = model(input_ids, use_cache=True)
    logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values

    generated_ids = input_ids.clone()

    # ── Decode: one token at a time, reusing the cache ───────────────
    for _ in range(max_new_tokens):
        # temperature scaling
        scaled_logits = logits
        if temperature != 1.0:
            scaled_logits = scaled_logits / temperature

        # top-k filtering
        if top_k > 0:
            top_k_values, _ = torch.topk(scaled_logits, top_k)
            min_top_k = top_k_values[:, -1].unsqueeze(-1)
            scaled_logits = torch.where(
                scaled_logits < min_top_k,
                torch.full_like(scaled_logits, float("-inf")),
                scaled_logits,
            )

        probs = torch.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        # Forward pass with ONLY the new token + cached K/V
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

    elapsed = time.perf_counter() - start

    num_tokens = generated_ids.shape[1] - input_ids.shape[1]
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return {
        "text": text,
        "num_tokens": num_tokens,
        "elapsed": elapsed,
        "tok_per_sec": num_tokens / elapsed if elapsed > 0 else 0.0,
    }
