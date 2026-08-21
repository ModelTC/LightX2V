"""Paged FlashAttention 3 support for single-token autoregressive decode."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate

try:
    from flash_attn_interface import flash_attn_with_kvcache as flash_attn3_with_kvcache
    from flash_attn_interface import get_scheduler_metadata as get_flash_attn3_scheduler_metadata
except ImportError:
    flash_attn3_with_kvcache = None
    get_flash_attn3_scheduler_metadata = None


def require_paged_flash_attn3() -> None:
    if flash_attn3_with_kvcache is None or get_flash_attn3_scheduler_metadata is None:
        raise ImportError("flash_attn3_paged requires the standalone FlashAttention 3 package with paged-KV and scheduler-metadata support.")


def build_flash_attn3_decode_scheduler_metadata(
    *,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    num_query_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    page_size: int,
    qkv_dtype: torch.dtype,
    max_num_splits: int,
) -> torch.Tensor:
    """Build FlashAttention 3 scheduling metadata for one-token decode."""

    return get_flash_attn3_scheduler_metadata(
        batch_size=int(cache_seqlens.numel()),
        max_seqlen_q=1,
        max_seqlen_k=int(max_seqlen_k),
        num_heads_q=int(num_query_heads),
        num_heads_kv=int(num_key_value_heads),
        headdim=int(head_dim),
        cache_seqlens=cache_seqlens,
        qkv_dtype=qkv_dtype,
        page_size=int(page_size),
        max_seqlen_k_new=0,
        causal=True,
        num_splits=int(max_num_splits),
    )


@triton.jit
def _paged_kv_store_kernel(
    key,
    value,
    key_cache,
    value_cache,
    page_table,
    cache_seqlens,
    key_stride_head: tl.constexpr,
    key_stride_dim: tl.constexpr,
    value_stride_head: tl.constexpr,
    value_stride_dim: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < num_kv_heads * head_dim
    head = offsets // head_dim
    dim = offsets - head * head_dim

    sequence_length = tl.load(cache_seqlens)
    token_position = sequence_length - 1
    logical_page = token_position // page_size
    page_offset = token_position - logical_page * page_size
    physical_page = tl.load(page_table + logical_page)
    cache_offset = ((physical_page * page_size + page_offset) * num_kv_heads + head) * head_dim + dim

    key_value = tl.load(key + head * key_stride_head + dim * key_stride_dim, mask=valid)
    value_value = tl.load(value + head * value_stride_head + dim * value_stride_dim, mask=valid)
    tl.store(key_cache + cache_offset, key_value, mask=valid)
    tl.store(value_cache + cache_offset, value_value, mask=valid)


def store_paged_kv(key, value, key_cache, value_cache, page_table, cache_seqlens) -> None:
    """Store one strided K/V token in an NHD paged cache."""

    num_kv_heads = int(key.shape[1])
    head_dim = int(key.shape[-1])
    total = num_kv_heads * head_dim
    _paged_kv_store_kernel[(triton.cdiv(total, 256),)](
        key,
        value,
        key_cache,
        value_cache,
        page_table,
        cache_seqlens,
        key_stride_head=key.stride(1),
        key_stride_dim=key.stride(3),
        value_stride_head=value.stride(1),
        value_stride_dim=value.stride(3),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=int(key_cache.shape[1]),
        BLOCK=256,
    )


@ATTN_WEIGHT_REGISTER("flash_attn3_paged")
class PagedFlashAttn3Weight(AttnWeightTemplate):
    """Native-GQA FlashAttention 3 decode over an NHD paged KV cache."""

    def __init__(self):
        require_paged_flash_attn3()
        self.config = {}

    def apply(self, q, k, v, **kwargs):
        raise RuntimeError("flash_attn3_paged is decode-only; use apply_decode with a persistent paged KV cache.")

    def apply_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        scheduler_metadata: torch.Tensor,
        max_num_splits: int,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        store_paged_kv(k, v, k_cache, v_cache, page_table, cache_seqlens)
        output = flash_attn3_with_kvcache(
            q.transpose(1, 2),
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=True,
            scheduler_metadata=scheduler_metadata,
            num_splits=int(max_num_splits),
        )
        return output.transpose(1, 2)


__all__ = [
    "PagedFlashAttn3Weight",
    "build_flash_attn3_decode_scheduler_metadata",
    "require_paged_flash_attn3",
    "store_paged_kv",
]
