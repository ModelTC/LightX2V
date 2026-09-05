from __future__ import annotations

import math

import torch
from loguru import logger

from lightx2v_platform.registry_factory import PLATFORM_SPARSE_OPERATOR_REGISTER

try:
    from moorcat._C import blocksparse as _blocksparse
except (ImportError, OSError) as exc:
    logger.info(f"Moorcat block-sparse attention is unavailable: {exc}")
    _blocksparse = None


_BLOCK_SIZE = 128
_RUNTIME_CACHE = {}
_VALIDATED_CONTRACTS = set()


def _runtime_tensors(
    q: torch.Tensor,
    seq_len: int,
    heads: int,
    q_blocks: int,
    topk: int,
):
    key = (q.device, seq_len, heads, q_blocks, topk)
    cached = _RUNTIME_CACHE.get(key)
    if cached is None:
        cached = (
            torch.arange(1, heads + 1, dtype=torch.int32, device=q.device),
            torch.full(
                (1, heads, q_blocks),
                topk,
                dtype=torch.int32,
                device=q.device,
            ),
            torch.tensor([0, seq_len], dtype=torch.int32, device=q.device),
        )
        _RUNTIME_CACHE[key] = cached
    return cached


def _validate_lut_once(
    block_indices: torch.Tensor,
    *,
    heads: int,
    q_blocks: int,
    kv_blocks: int,
    topk: int,
) -> None:
    expected = (1, heads, q_blocks, topk)
    if tuple(block_indices.shape) != expected:
        raise ValueError(f"Moorcat Q128 LUT must have shape {expected}, got {tuple(block_indices.shape)}")
    contract = (expected, kv_blocks, block_indices.device)
    if contract in _VALIDATED_CONTRACTS:
        return
    min_index = int(block_indices.min().item())
    max_index = int(block_indices.max().item())
    if min_index < 0 or max_index >= kv_blocks:
        raise ValueError(f"Moorcat Q128 LUT indices must be in [0, {kv_blocks}), got [{min_index}, {max_index}]")
    ordered = torch.sort(block_indices, dim=-1).values
    if not bool(torch.all(ordered[..., 1:] != ordered[..., :-1]).item()):
        raise ValueError("Moorcat Q128 LUT rows must contain distinct K blocks")
    _VALIDATED_CONTRACTS.add(contract)


@torch.library.custom_op(
    "lightx2v::musa_moorcat_sparse",
    mutates_args=(),
    device_types="cuda",
)
def musa_moorcat_sparse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    topk: int,
    softmax_scale: float,
    validate_lut: bool,
) -> torch.Tensor:
    if _blocksparse is None:
        raise RuntimeError("Moorcat is not importable. Install the Moorcat package supplied with the official MUSA SGL image.")

    seq_len, heads, _ = q.shape
    q_blocks = math.ceil(seq_len / _BLOCK_SIZE)
    kv_blocks = math.ceil(k.shape[0] / _BLOCK_SIZE)
    if validate_lut:
        _validate_lut_once(
            block_indices,
            heads=heads,
            q_blocks=q_blocks,
            kv_blocks=kv_blocks,
            topk=topk,
        )

    indices = block_indices.to(dtype=torch.int32).contiguous()
    head_mask_type, counts, cu_seqlens = _runtime_tensors(q, seq_len, heads, q_blocks, topk)

    padded_kv_len = kv_blocks * _BLOCK_SIZE
    if padded_kv_len != k.shape[0]:
        k, v, _ = _blocksparse._pad_kv_tail_storage_to_tile128(k, v, padded_kv_len)

    out, _ = _blocksparse.block_sparse_attn_fwd_indexed(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        head_mask_type=head_mask_type,
        streaming_info=None,
        q2k_block_sparse_index=indices,
        q2k_block_sparse_num=counts,
        max_kv_blocks_per_q=topk,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        softmax_scale=softmax_scale,
        is_causal=False,
        block_size=_BLOCK_SIZE,
    )
    return out


@musa_moorcat_sparse.register_fake
def _musa_moorcat_sparse_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    topk: int,
    softmax_scale: float,
    validate_lut: bool,
) -> torch.Tensor:
    return torch.empty_like(q)


@PLATFORM_SPARSE_OPERATOR_REGISTER("musa_moorcat_sparse")
class MusaMoorcatSparseOperator:
    """Run a Q128/K128 block map with Moorcat's BF16 sparse kernel."""

    q_block_size = _BLOCK_SIZE
    k_block_size = _BLOCK_SIZE
    block_indices_only = True

    def __init__(self, operator_setting=None):
        setting = dict(operator_setting or {})
        self.topk = int(setting.get("topk", 16))
        self.center_k = bool(setting.get("center_k", True))
        self.validate_lut = bool(setting.get("validate_lut", setting.get("validate_mask", True)))
        if self.topk <= 0:
            raise ValueError(f"Moorcat topk must be positive, got {self.topk}")

    @torch.compiler.disable
    def __call__(
        self,
        q,
        k,
        v,
        mask,
        block_indices=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        if _blocksparse is None:
            raise RuntimeError("The official Moorcat binary is unavailable. Install the Moorcat package supplied with the official MUSA SGL image.")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("Moorcat requires flattened [tokens, heads, head_dim] Q/K/V")
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(f"Moorcat supports self-attention only, got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}")
        if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("Moorcat requires BF16 Q/K/V")
        if q.shape[-1] != 128:
            raise ValueError(f"Moorcat requires head_dim=128, got {q.shape[-1]}")
        if bool(kwargs.get("causal", False)):
            raise ValueError("Moorcat supports non-causal attention only")
        if cu_seqlens_q is not None and cu_seqlens_q.numel() != 2:
            raise ValueError("Moorcat supports one packed sequence only")
        if cu_seqlens_kv is not None and cu_seqlens_kv.numel() != 2:
            raise ValueError("Moorcat supports one packed sequence only")
        if block_indices is None:
            raise ValueError("Moorcat requires Q128/K128 block indices")

        softmax_scale = kwargs.get("softmax_scale")
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        out = musa_moorcat_sparse(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            block_indices,
            self.topk,
            float(softmax_scale),
            self.validate_lut,
        )
        return out.reshape(q.shape[0], -1)
