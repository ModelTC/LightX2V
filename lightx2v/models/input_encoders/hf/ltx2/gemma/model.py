"""Local Gemma 3 entry point with MLU Flash Attention support."""

from typing import Any

import torch
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForConditionalGeneration as HFGemma3ForConditionalGeneration,
)

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

MLU_FLASH_ATTN = "mlu_flash_attn"
_platform_attention_ops: dict[str, Any] = {}


def _get_platform_attention_op(attn_implementation: str):
    if attn_implementation not in _platform_attention_ops:
        op_cls = ATTN_WEIGHT_REGISTER.get(attn_implementation)
        if op_cls is None:
            raise RuntimeError(f"`{attn_implementation}` is not registered for the active platform. For MLU, set PLATFORM=cambricon_mlu and install torch_mlu_ops.")
        _platform_attention_ops[attn_implementation] = op_cls()
    return _platform_attention_ops[attn_implementation]


def _prepare_attn_bias(
    attention_mask: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_heads, query_length, _ = query.shape
    key_length = key.shape[-2]
    attn_bias = attention_mask[..., :query_length, :key_length]

    if attn_bias.dtype == torch.bool:
        attn_bias = torch.where(
            attn_bias,
            torch.zeros((), dtype=query.dtype, device=query.device),
            torch.full((), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device),
        )
    else:
        attn_bias = attn_bias.to(device=query.device, dtype=query.dtype)

    if attn_bias.shape[0] not in (1, batch_size) or attn_bias.shape[1] not in (1, num_heads):
        raise ValueError(f"Gemma 3 MLU attention mask must be broadcastable to {(batch_size, num_heads, query_length, key_length)}, got {tuple(attn_bias.shape)}.")

    # torch_mlu_ops requires a dense [B, Hq, Q, K] bias. It produces NaNs
    # with the bfloat16 finfo.min sentinel at Gemma's real attention shape,
    # so normalize masked entries to a finite value first. Fully masked
    # left-padding query rows are irrelevant and must be opened as well.
    attn_bias = attn_bias.expand(batch_size, num_heads, query_length, key_length).contiguous()
    masked_positions = attn_bias < -1_000
    fully_masked_rows = masked_positions.all(dim=-1, keepdim=True)
    attn_bias.masked_fill_(masked_positions, -10_000)
    return attn_bias.masked_fill_(fully_masked_rows, 0)


def mlu_flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Transformers attention callback backed by ``torch_mlu_ops``."""
    if dropout:
        raise NotImplementedError("Gemma 3 `mlu_flash_attn` only supports inference with attention dropout disabled.")
    if kwargs.get("output_attentions", False):
        raise NotImplementedError("Gemma 3 `mlu_flash_attn` does not return attention weights.")

    softcap = softcap if softcap is not None else getattr(module, "attn_logit_softcapping", None)
    if softcap is not None:
        raise NotImplementedError("Gemma 3 `mlu_flash_attn` does not support attention logit softcapping.")

    batch_size, num_heads, query_length, head_dim = query.shape
    key_length = key.shape[-2]
    attn_bias = None
    is_causal = kwargs.get("is_causal", getattr(module, "is_causal", False))
    window_size_left = -1
    window_size_right = -1

    if attention_mask is not None:
        attn_bias = _prepare_attn_bias(attention_mask, query, key)
        # The additive bias already contains causal, sliding-window, padding,
        # and Gemma image-token masking.
        is_causal = False
    elif sliding_window is not None and key_length > sliding_window:
        window_size_left = sliding_window - 1
        window_size_right = 0 if is_causal else sliding_window - 1
    elif query_length == 1:
        # A single decode query must see the complete existing KV cache.
        is_causal = False

    output = _get_platform_attention_op(MLU_FLASH_ATTN).apply(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        max_seqlen_q=query_length,
        max_seqlen_kv=key_length,
        softmax_scale=scaling,
        causal=is_causal,
        attn_bias=attn_bias,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    return output.view(batch_size, query_length, num_heads, head_dim), None


def register_gemma_attention_backend(attn_implementation: str) -> None:
    """Register a platform attention backend with Transformers' Gemma interface."""
    if attn_implementation != MLU_FLASH_ATTN:
        raise ValueError(f"Unsupported Gemma attention backend: {attn_implementation!r}. Currently only {MLU_FLASH_ATTN!r} is supported.")
    # Resolve and instantiate before loading model weights so missing platform
    # dependencies fail early instead of at the first forward pass.
    _get_platform_attention_op(attn_implementation)

    ALL_ATTENTION_FUNCTIONS.register(attn_implementation, mlu_flash_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(attn_implementation, sdpa_mask)


class Gemma3ForConditionalGeneration(HFGemma3ForConditionalGeneration):
    """Project-local Gemma 3 model supporting registered platform attention."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        requested_implementation = kwargs.get("attn_implementation")
        text_implementation = requested_implementation.get("text_config") if isinstance(requested_implementation, dict) else requested_implementation
        if text_implementation == MLU_FLASH_ATTN:
            register_gemma_attention_backend(text_implementation)
            if isinstance(requested_implementation, str):
                # A scalar recursively affects Gemma 3's SigLIP tower. Platform
                # backends configured for Gemma text must stay text-only.
                kwargs["attn_implementation"] = {
                    "text_config": text_implementation,
                }
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


__all__ = [
    "Gemma3ForConditionalGeneration",
    "mlu_flash_attention_forward",
    "register_gemma_attention_backend",
]
