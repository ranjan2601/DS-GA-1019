"""
Inference module — baseline text generation.

This is the unoptimized baseline. Each optimization phase will either
modify this module or provide an alternative generate function with the
same signature so benchmarking stays consistent.
"""

import time

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.config import MAX_NEW_TOKENS, TEMPERATURE, TOP_K


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
) -> dict:
    """
    Baseline autoregressive generation using HuggingFace's built-in generate.

    Returns a dict with:
        text        – the generated string (prompt + continuation)
        num_tokens  – number of new tokens generated
        elapsed     – wall-clock seconds for the generation loop
        tok_per_sec – tokens per second
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    start = time.perf_counter()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - start

    generated_ids = output_ids[0, input_ids.shape[1]:]
    num_tokens = len(generated_ids)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {
        "text": text,
        "num_tokens": num_tokens,
        "elapsed": elapsed,
        "tok_per_sec": num_tokens / elapsed if elapsed > 0 else 0.0,
    }


@torch.no_grad()
def generate_manual(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
) -> dict:
    """
    Manual autoregressive loop — exposes the per-token forward pass
    so we can profile individual steps (attention, matmul, sampling).

    This is the version we will optimize in later phases.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated_ids = input_ids.clone()

    start = time.perf_counter()
    for _ in range(max_new_tokens):
        outputs = model(generated_ids)
        # logits for the last token: shape (1, vocab_size)
        logits = outputs.logits[:, -1, :]

        # temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # top-k filtering
        if top_k > 0:
            top_k_values, _ = torch.topk(logits, top_k)
            min_top_k = top_k_values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_top_k,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    elapsed = time.perf_counter() - start

    num_tokens = generated_ids.shape[1] - input_ids.shape[1]
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return {
        "text": text,
        "num_tokens": num_tokens,
        "elapsed": elapsed,
        "tok_per_sec": num_tokens / elapsed if elapsed > 0 else 0.0,
    }
