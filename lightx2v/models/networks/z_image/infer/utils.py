from typing import Tuple

import torch

try:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
except ImportError:
    apply_rope_with_cos_sin_cache_inplace = None


def apply_wan_rope_with_flashinfer(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos_sin_cache: torch.Tensor,
):
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


def apply_rotary_emb_qwen(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_rotated = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)).squeeze(0)
    xk_rotated = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)).squeeze(0)
    freqs_cis = cos_sin_cache.unsqueeze(1)
    xq_out = torch.view_as_real(xq_rotated * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_rotated * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def patchify(hidden_states: torch.Tensor, patch_size: int = 2, f_patch_size: int = 1) -> torch.Tensor:
    B, C, H, W = hidden_states.shape
    pH = pW = patch_size
    pF = f_patch_size
    F = 1
    F_tokens = F // pF
    H_tokens = H // pH
    W_tokens = W // pW

    hidden_states = hidden_states.view(B, C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
    hidden_states = hidden_states.permute(0, 2, 4, 6, 3, 5, 7, 1)
    hidden_states = hidden_states.reshape(B, F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

    return hidden_states
