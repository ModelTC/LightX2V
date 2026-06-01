from typing import Tuple

import torch

try:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
except ImportError:
    apply_rope_with_cos_sin_cache_inplace = None

try:
    from magi_compiler import magi_register_custom_op
except ImportError:
    magi_register_custom_op = None

# Re-export for transformer_infer init.
from lightx2v.common.magi_custom_op_mode import (
    set_magi_custom_op_mode,  # noqa: F401
    use_magi_custom_ops,
)


def _qwen_rope_meta(xq, xk, cos_sin_cache):
    return torch.empty_like(xq), torch.empty_like(xk)


def _apply_qwen_rope_with_flashinfer_eager(xq, xk, cos_sin_cache):
    L, H, D = xq.shape

    query = xq.reshape(L, H * D).contiguous()
    key = xk.reshape(L, H * D).contiguous()

    positions = torch.arange(L, device="cpu", dtype=torch.long).to(xq.device, non_blocking=True)

    apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=query,
        key=key,
        head_size=D,
        cos_sin_cache=cos_sin_cache,
        is_neox=False,
    )

    xq_out = query.view(L, H, D)
    xk_out = key.view(L, H, D)
    return xq_out, xk_out


def _apply_qwen_rope_with_flashinfer_magi(xq, xk, cos_sin_cache):
    L, H, D = xq.shape

    query = xq.reshape(L, H * D).contiguous().clone()
    key = xk.reshape(L, H * D).contiguous().clone()

    positions = torch.arange(L, device="cpu", dtype=torch.long).to(xq.device, non_blocking=True)

    apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=query,
        key=key,
        head_size=D,
        cos_sin_cache=cos_sin_cache,
        is_neox=False,
    )

    xq_out = query.view(L, H, D)
    xk_out = key.view(L, H, D)
    return xq_out, xk_out


def _apply_qwen_rope_with_torch_impl(xq, xk, cos_sin_cache):
    xq_rotated = torch.view_as_complex(xq.float().unflatten(-1, (-1, 2)))
    xk_rotated = torch.view_as_complex(xk.float().unflatten(-1, (-1, 2)))
    freqs_cis = cos_sin_cache.unsqueeze(1)
    xq_out = torch.view_as_real(xq_rotated * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_rotated * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


if magi_register_custom_op is not None and apply_rope_with_cos_sin_cache_inplace is not None:

    @magi_register_custom_op(
        "lightx2v::qwen_rope_flashinfer",
        infer_output_meta_fn=_qwen_rope_meta,
        is_subgraph_boundary=True,
    )
    def _qwen_rope_flashinfer_custom_op(xq: torch.Tensor, xk: torch.Tensor, cos_sin_cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _apply_qwen_rope_with_flashinfer_magi(xq, xk, cos_sin_cache)


if magi_register_custom_op is not None:

    @magi_register_custom_op(
        "lightx2v::qwen_rope_torch",
        infer_output_meta_fn=_qwen_rope_meta,
        is_subgraph_boundary=True,
    )
    def _qwen_rope_torch_custom_op(xq: torch.Tensor, xk: torch.Tensor, cos_sin_cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _apply_qwen_rope_with_torch_impl(xq, xk, cos_sin_cache)


def apply_qwen_rope_with_flashinfer(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos_sin_cache: torch.Tensor,
):
    if use_magi_custom_ops() and magi_register_custom_op is not None and apply_rope_with_cos_sin_cache_inplace is not None:
        return torch.ops.lightx2v.qwen_rope_flashinfer(xq, xk, cos_sin_cache)
    return _apply_qwen_rope_with_flashinfer_eager(xq, xk, cos_sin_cache)


def apply_qwen_rope_with_torch(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_magi_custom_ops() and magi_register_custom_op is not None:
        return torch.ops.lightx2v.qwen_rope_torch(xq, xk, cos_sin_cache)
    return _apply_qwen_rope_with_torch_impl(xq, xk, cos_sin_cache)


def apply_qwen_rope_with_torch_naive(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos_sin_cache.real.unsqueeze(1)
    sin = cos_sin_cache.imag.unsqueeze(1)

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        x_out = torch.empty_like(x)
        x_out[..., 0::2] = x_rot_even
        x_out[..., 1::2] = x_rot_odd
        return x_out

    xq_out = _rotate(xq)
    xk_out = _rotate(xk)
    return xq_out, xk_out
