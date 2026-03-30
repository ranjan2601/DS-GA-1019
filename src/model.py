"""
Model loading and management.

Handles downloading and loading the LLM and tokenizer from HuggingFace.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MODEL_NAME


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """
    Load a HuggingFace causal LM and its tokenizer.

    Returns:
        (model, tokenizer) tuple. Model is in eval mode on CPU with float32.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    model.eval()

    # GPT-2 has no pad token by default — use eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
