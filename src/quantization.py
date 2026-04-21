"""
Optimization 2b: INT8 Quantization.

Bottleneck addressed: FP32 weight matrices (124M params × 4 bytes = ~475 MB)
consume memory bandwidth and slow down matrix multiplications. Reducing to
INT8 (1 byte per weight) cuts memory 4× and enables faster integer matmul.

We implement quantization manually (not using an off-the-shelf library)
to demonstrate the technique with course tools (NumPy, Numba).
"""

import time
import copy

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.config import MAX_NEW_TOKENS, TEMPERATURE, TOP_K

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ── Quantization helpers ─────────────────────────────────────────────────────


def quantize_tensor_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """
    Symmetric per-tensor INT8 quantization.

    Maps FP32 values to [-127, 127] range using a single scale factor.
    Returns (quantized_int8_tensor, scale, zero_point).
    """
    fp32 = tensor.float()
    max_abs = fp32.abs().max().item()
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    quantized = torch.clamp(torch.round(fp32 / scale), -127, 127).to(torch.int8)
    return quantized, scale, 0.0


def dequantize_tensor(quantized: torch.Tensor, scale: float) -> torch.Tensor:
    """Dequantize INT8 back to FP32."""
    return quantized.float() * scale


if NUMBA_AVAILABLE:

    @njit(parallel=True, cache=True)
    def _int8_matmul_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        INT8 matrix multiplication using Numba JIT.

        a: (M, K) int8
        b: (K, N) int8
        returns: (M, N) int32 accumulator
        """
        M, K = a.shape
        N = b.shape[1]
        out = np.zeros((M, N), dtype=np.int32)
        for i in prange(M):
            for j in range(N):
                acc = np.int32(0)
                for k in range(K):
                    acc += np.int32(a[i, k]) * np.int32(b[k, j])
                out[i, j] = acc
        return out


def int8_matmul(
    a_quant: torch.Tensor,
    a_scale: float,
    b_quant: torch.Tensor,
    b_scale: float,
) -> torch.Tensor:
    """
    Perform matmul on quantized INT8 tensors, return FP32 result.

    Uses Numba JIT if available, falls back to PyTorch int32 matmul.
    """
    if NUMBA_AVAILABLE:
        a_np = a_quant.numpy().astype(np.int8)
        b_np = b_quant.numpy().astype(np.int8)
        result_np = _int8_matmul_numba(a_np, b_np)
        result = torch.from_numpy(result_np).float()
    else:
        # Fallback: cast to int32 for matmul (no int8 matmul in PyTorch CPU)
        result = torch.matmul(
            a_quant.to(torch.int32), b_quant.to(torch.int32)
        ).float()

    return result * (a_scale * b_scale)


# ── Model quantization ───────────────────────────────────────────────────────


class QuantizedLinear(torch.nn.Module):
    """
    Drop-in replacement for nn.Linear that stores INT8 weights
    and dequantizes on the fly during forward pass.

    Only the INT8 weights (1 byte each) and a scalar scale are stored,
    plus the original FP32 bias (which is tiny). The original FP32 weight
    matrix is discarded, giving ~4× weight memory reduction.
    """

    def __init__(self, weight_data: torch.Tensor, bias_data: torch.Tensor | None, transposed: bool = False):
        super().__init__()
        # If weight is (in_features, out_features) like Conv1D, transpose it
        # to (out_features, in_features) for F.linear
        if transposed:
            weight_data = weight_data.T.contiguous()
        weight_quant, scale, _ = quantize_tensor_int8(weight_data)
        self.register_buffer("weight_quantized", weight_quant)
        self.register_buffer("scale", torch.tensor(scale))
        if bias_data is not None:
            self.register_buffer("bias", bias_data.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_fp32 = self.weight_quantized.float() * self.scale.item()
        return torch.nn.functional.linear(x, weight_fp32, self.bias)


def quantize_model(model: PreTrainedModel) -> PreTrainedModel:
    """
    Replace all nn.Linear AND Conv1D layers with QuantizedLinear.

    GPT-2 uses Conv1D (from transformers) for most weight layers,
    so we must handle both. Returns a new model (does not modify original).
    """
    quantized_model = copy.deepcopy(model)
    _replace_quantizable_layers(quantized_model)
    quantized_model.eval()
    return quantized_model


def _replace_quantizable_layers(module: torch.nn.Module):
    """Recursively replace nn.Linear and Conv1D with QuantizedLinear."""
    from transformers.pytorch_utils import Conv1D

    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, QuantizedLinear(
                child.weight.data, child.bias.data if child.bias is not None else None,
                transposed=False,
            ))
        elif isinstance(child, Conv1D):
            # Conv1D stores weight as (in_features, out_features) — transposed
            setattr(module, name, QuantizedLinear(
                child.weight.data, child.bias.data if child.bias is not None else None,
                transposed=True,
            ))
        else:
            _replace_quantizable_layers(child)


def get_model_size_mb(model: PreTrainedModel) -> float:
    """Total model memory in MB (params + buffers), accounting for dtype."""
    param_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    buffer_bytes = sum(
        b.numel() * b.element_size() for b in model.buffers()
    )
    return (param_bytes + buffer_bytes) / (1024 ** 2)


def get_model_size_breakdown(model: PreTrainedModel) -> dict:
    """Detailed breakdown of model memory by params vs buffers and dtype."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    # Count by dtype
    dtype_breakdown = {}
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        dtype_str = str(tensor.dtype)
        if dtype_str not in dtype_breakdown:
            dtype_breakdown[dtype_str] = {"count": 0, "bytes": 0}
        dtype_breakdown[dtype_str]["count"] += tensor.numel()
        dtype_breakdown[dtype_str]["bytes"] += tensor.numel() * tensor.element_size()

    return {
        "param_mb": param_bytes / (1024 ** 2),
        "buffer_mb": buffer_bytes / (1024 ** 2),
        "total_mb": (param_bytes + buffer_bytes) / (1024 ** 2),
        "dtype_breakdown": {
            k: {
                "elements": v["count"],
                "mb": v["bytes"] / (1024 ** 2),
            }
            for k, v in dtype_breakdown.items()
        },
    }


# ── Generation with quantized model ─────────────────────────────────────────


@torch.no_grad()
def generate_quantized(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
) -> dict:
    """
    Autoregressive generation using a quantized model + KV-cache.

    Combines INT8 quantization with KV-caching for maximum benefit.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    start = time.perf_counter()

    # Prefill with cache
    outputs = model(input_ids, use_cache=True)
    logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values

    generated_ids = input_ids.clone()

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
        next_token = torch.multinomial(probs, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

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
