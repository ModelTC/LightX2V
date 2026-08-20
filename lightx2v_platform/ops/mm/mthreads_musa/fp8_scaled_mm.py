"""MUSA FP8 per-token/per-channel scaled matrix multiplication.

This module intentionally does not depend on DeepGEMM or sgl-kernel.  It uses
the row-wise scale support exposed by ``torch_musa`` through
``torch._scaled_mm``:

* activations are quantized per token and use scales shaped ``[M, 1]``;
* weights are quantized per output channel and use scales shaped ``[N, 1]``;
* the scaled matrix multiplication consumes the weight matrix as ``[K, N]``.
"""

from __future__ import annotations

import torch

try:
    from vllm import _custom_ops as vllm_ops
except ImportError:  # pragma: no cover - exercised only without vLLM installed
    vllm_ops = None


FP8_DTYPE = torch.float8_e4m3fn
_OUTPUT_DTYPES = (torch.bfloat16, torch.float16)


def _check_2d(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(tensor.shape)}")


def _normalize_token_scale(scale: torch.Tensor, rows: int) -> torch.Tensor:
    if scale.numel() != rows:
        raise ValueError(f"activation scale must contain one value per token ({rows}), got shape {tuple(scale.shape)}")
    return scale.reshape(rows, 1)


def _normalize_channel_scale(scale: torch.Tensor, columns: int) -> torch.Tensor:
    if scale.numel() != columns:
        raise ValueError(f"weight scale must contain one value per output channel ({columns}), got shape {tuple(scale.shape)}")
    return scale.reshape(1, columns)


def per_token_quant_fp8(
    input_tensor: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize a 2D activation tensor to FP8 per token.

    Args:
        input_tensor: Floating-point activation matrix shaped ``[M, K]``.
        output: Optional preallocated FP8 output shaped ``[M, K]``.
        eps: Minimum absolute maximum used by the PyTorch fallback.

    Returns:
        ``(quantized, scale)`` where scale is FP32 and shaped ``[M, 1]``.
    """

    _check_2d("input_tensor", input_tensor)
    if not input_tensor.is_floating_point():
        raise TypeError(f"input_tensor must be floating point, got {input_tensor.dtype}")
    if input_tensor.device.type != "musa":
        raise ValueError(f"input_tensor must be on MUSA, got {input_tensor.device}")

    if output is not None:
        if output.shape != input_tensor.shape:
            raise ValueError(f"output shape {tuple(output.shape)} does not match input shape {tuple(input_tensor.shape)}")
        if output.dtype != FP8_DTYPE or output.device != input_tensor.device:
            raise ValueError(f"output must use {FP8_DTYPE} on {input_tensor.device}, got {output.dtype} on {output.device}")

    if vllm_ops is not None:
        quantized, scale = vllm_ops.scaled_fp8_quant(
            input_tensor,
            scale=None,
            scale_ub=None,
            use_per_token_if_dynamic=True,
            output=output,
        )
        return quantized, scale.reshape(input_tensor.shape[0], 1).float()

    fp8_max = torch.finfo(FP8_DTYPE).max
    scale = input_tensor.abs().float().amax(dim=-1, keepdim=True).clamp_min(eps) / fp8_max
    quantized = (input_tensor.float() / scale).clamp(-fp8_max, fp8_max).to(FP8_DTYPE)
    if output is not None:
        output.copy_(quantized)
        quantized = output
    return quantized, scale


def fp8_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: torch.Tensor | None = None,
    use_fast_accum: bool = True,
) -> torch.Tensor:
    """Multiply pre-quantized FP8 matrices using per-token/channel scales.

    ``mat_a`` is shaped ``[M, K]`` and ``mat_b`` is shaped ``[K, N]``.
    ``scale_a`` accepts ``[M]`` or ``[M, 1]``; ``scale_b`` accepts ``[N]``,
    ``[N, 1]``, or ``[1, N]``.  Scales must be FP32 because that is the native
    muDNN row-wise scaled-MM contract.
    """

    _check_2d("mat_a", mat_a)
    _check_2d("mat_b", mat_b)
    if mat_a.shape[1] != mat_b.shape[0]:
        raise ValueError(f"incompatible matrix shapes: {tuple(mat_a.shape)} and {tuple(mat_b.shape)}")
    if mat_a.dtype != FP8_DTYPE or mat_b.dtype != FP8_DTYPE:
        raise TypeError(f"mat_a and mat_b must both use {FP8_DTYPE}")
    if mat_a.device != mat_b.device or mat_a.device.type != "musa":
        raise ValueError(f"mat_a and mat_b must be on the same MUSA device, got {mat_a.device} and {mat_b.device}")
    if scale_a.dtype != torch.float32 or scale_b.dtype != torch.float32:
        raise TypeError("scale_a and scale_b must use torch.float32")
    if scale_a.device != mat_a.device or scale_b.device != mat_a.device:
        raise ValueError("matrices and scales must be on the same MUSA device")
    if out_dtype not in _OUTPUT_DTYPES:
        raise TypeError(f"out_dtype must be bfloat16 or float16, got {out_dtype}")

    rows, columns = mat_a.shape[0], mat_b.shape[1]
    scale_a_2d = _normalize_token_scale(scale_a, rows)
    scale_b_2d = _normalize_channel_scale(scale_b, columns)

    if bias is not None:
        if bias.ndim != 1 or bias.numel() != columns:
            raise ValueError(f"bias must have shape ({columns},), got {tuple(bias.shape)}")
        if bias.device != mat_a.device:
            raise ValueError("bias and matrices must be on the same MUSA device")

    output = torch._scaled_mm(
        mat_a,
        mat_b,
        scale_a=scale_a_2d,
        scale_b=scale_b_2d,
        bias=bias,
        out_dtype=out_dtype,
        use_fast_accum=use_fast_accum,
    )
    # PyTorch before 2.5 returned ``(output, amax)``.
    return output[0] if isinstance(output, tuple) else output


def fp8_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = True,
) -> torch.Tensor:
    """Apply a PyTorch-layout FP8 linear layer with dynamic token scaling.

    Args:
        input_tensor: Activation tensor shaped ``[..., K]``.
        weight: Per-channel quantized FP8 weight shaped ``[N, K]``.
        weight_scale: FP32 weight scale shaped ``[N]`` or ``[N, 1]``.
        bias: Optional bias shaped ``[N]``.
        out_dtype: BF16 or FP16 output. Defaults to the input dtype when valid,
            otherwise BF16.
    """

    if input_tensor.ndim < 2:
        raise ValueError(f"input_tensor must have at least 2 dimensions, got {tuple(input_tensor.shape)}")
    _check_2d("weight", weight)
    if input_tensor.shape[-1] != weight.shape[1]:
        raise ValueError(f"input hidden size {input_tensor.shape[-1]} does not match weight hidden size {weight.shape[1]}")

    output_dtype = out_dtype or (input_tensor.dtype if input_tensor.dtype in _OUTPUT_DTYPES else torch.bfloat16)
    input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
    input_quant, input_scale = per_token_quant_fp8(input_2d)
    output_2d = fp8_scaled_mm(
        input_quant,
        weight.t(),
        input_scale,
        weight_scale,
        out_dtype=output_dtype,
        bias=bias,
        use_fast_accum=use_fast_accum,
    )
    return output_2d.reshape(*input_tensor.shape[:-1], weight.shape[0])


__all__ = ["FP8_DTYPE", "fp8_linear", "fp8_scaled_mm", "per_token_quant_fp8"]
