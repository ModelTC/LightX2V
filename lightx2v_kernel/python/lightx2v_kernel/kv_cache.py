# SPDX-License-Identifier: Apache-2.0
"""Fused NVFP4 KV-cache dequant (parallel CUDA)."""

import torch


def _dtype_to_code(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float16:
        return 1
    if dtype == torch.float32:
        return 2
    raise ValueError(f"Unsupported fused KV dequant dtype: {dtype}")


def dequantize_kv_cache_fp4(
    values: list[torch.Tensor],
    scale_factors: list[torch.Tensor],
    amax: list[torch.Tensor],
    *,
    num_heads: int,
    block_token_size: int,
    dtype: torch.dtype,
    e2m1_max: float,
    e4m3_max: float,
) -> torch.Tensor:
    """Dequantize multiple AR KV-cache blocks with one CUDA launch."""
    return torch.ops.lightx2v_kernel.dequantize_kv_cache_fp4.default(
        values,
        scale_factors,
        amax,
        num_heads,
        block_token_size,
        _dtype_to_code(dtype),
        e2m1_max,
        e4m3_max,
    )
