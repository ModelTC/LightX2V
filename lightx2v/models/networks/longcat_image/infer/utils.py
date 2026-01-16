from typing import Tuple

import torch

try:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
except ImportError:
    apply_rope_with_cos_sin_cache_inplace = None


def apply_longcat_rope_with_flashinfer(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos_sin_cache: torch.Tensor,
):
    """Apply rotary position embedding using flashinfer.

    Args:
        xq: Query tensor [L, H, D]
        xk: Key tensor [L, H, D]
        cos_sin_cache: Cosine and sine cache [L, D] where first half is cos, second half is sin

    Returns:
        Tuple of (xq, xk) with rotary embedding applied
    """
    L, H, D = xq.shape

    query = xq.reshape(L, H * D).contiguous()
    key = xk.reshape(L, H * D).contiguous()

    # Create positions directly on GPU to avoid CPU-GPU sync
    positions = torch.arange(L, device=xq.device, dtype=torch.long)

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


def apply_longcat_rope_with_torch(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding using PyTorch.

    Follows the diffusers implementation for LongCat/Flux.

    Args:
        xq: Query tensor [L, H, D]
        xk: Key tensor [L, H, D]
        freqs_cis: Tuple of (cos, sin) each [L, D]

    Returns:
        Tuple of (xq, xk) with rotary embedding applied
    """
    cos, sin = freqs_cis  # [L, D]

    # Expand for heads: [L, D] -> [L, 1, D]
    cos = cos[:, None, :]
    sin = sin[:, None, :]

    def _apply_rope(x, cos, sin):
        # Split into real and imaginary parts (interleaved format)
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [L, H, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out

    xq_out = _apply_rope(xq, cos, sin)
    xk_out = _apply_rope(xk, cos, sin)
    return xq_out, xk_out
