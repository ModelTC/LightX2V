"""Student-only sparse attention for MiniMax-H3 DMD training.

The block router and sparse softmax follow ``thu-ml/SLA``.  This adaptation
intentionally implements only the selected-block softmax branch ``o_s``: the
feature-map linear-attention branch ``o_l`` and its projection do not exist.
Consequently the processor is parameter-free and can be enabled for an
existing LoRA checkpoint without changing its state-dict layout.

Upstream implementation (Apache-2.0):
https://github.com/thu-ml/SLA/tree/main/sparse_linear_attention
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Mapping

import torch
from loguru import logger

_LOGGED_ROUTING_LAYOUTS: set[tuple[int, int, int, int]] = set()


def _apply_rotary_emb(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply the rotary convention used by Diffusers' MiniMax-H3."""
    rotary_dim = cos.shape[-1]
    rotary = hidden_states[..., :rotary_dim]
    passthrough = hidden_states[..., rotary_dim:]
    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    x1, x2 = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return torch.cat((rotary * cos + rotated * sin, passthrough), dim=-1).contiguous()


def _block_mean(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean-pool ``[B,H,L,D]`` into blocks, including a short final block."""
    batch, heads, length, dim = x.shape
    full_blocks, remainder = divmod(length, block_size)
    chunks = []
    if full_blocks:
        prefix = x[:, :, : full_blocks * block_size]
        prefix = prefix.reshape(batch, heads, full_blocks, block_size, dim)
        chunks.append(prefix.mean(dim=-2))
    if remainder:
        chunks.append(x[:, :, full_blocks * block_size :].mean(dim=-2, keepdim=True))
    if not chunks:
        raise ValueError("SLA requires a non-empty sequence.")
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=-2)


def retained_key_blocks(sequence_length: int, block_k: int, keep_ratio: float) -> tuple[int, int]:
    """Return ``(selected, total)`` block counts using SLA's floor rule."""
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}.")
    total = ceil(sequence_length / block_k)
    # Upstream uses int(ratio * K). Clamp to one so short sequences cannot
    # produce an empty LUT and NaNs in the online-softmax kernel.
    selected = max(1, min(total, floor(keep_ratio * total)))
    return selected, total


@torch.no_grad()
def _get_block_map(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    keep_ratio: float,
    block_q: int,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build SLA's per-head, per-query-block key-block routing table."""
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("SLA routing expects query/key in [B,H,L,D] layout.")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError(f"SLA query/key batch, heads and head_dim must match, got {query.shape} and {key.shape}.")
    if query.shape[-2] != key.shape[-2]:
        raise ValueError("The bundled SLA kernel currently supports self-attention with equal Q/K lengths only.")

    # Smooth-K and block-mean router from SLA utils.py.
    smooth_key = key - key.mean(dim=-2, keepdim=True)
    pooled_query = _block_mean(query, block_q)
    pooled_key = _block_mean(smooth_key, block_k)
    block_scores = pooled_query @ pooled_key.transpose(-1, -2)

    topk, total_key_blocks = retained_key_blocks(key.shape[-2], block_k, keep_ratio)
    if total_key_blocks != block_scores.shape[-1]:
        raise RuntimeError("Internal SLA key-block count mismatch.")
    lut = torch.topk(block_scores, topk, dim=-1, sorted=False).indices.contiguous()
    sparse_map = torch.zeros_like(block_scores, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map.contiguous(), lut, topk


@dataclass(frozen=True)
class MiniMaxH3StudentSLAConfig:
    enabled: bool = False
    sparsity: float = 0.85
    block_q: int = 64
    block_k: int = 64
    compute_dtype: torch.dtype = torch.bfloat16

    @property
    def keep_ratio(self) -> float:
        return 1.0 - self.sparsity

    def checkpoint_metadata(self) -> dict:
        return {
            "enabled": self.enabled,
            "sparsity": self.sparsity,
            "block_q": self.block_q,
            "block_k": self.block_k,
            "compute_dtype": str(self.compute_dtype).removeprefix("torch."),
            "linear_branch": False,
            "apply_to": "transformer_blocks",
        }

    @classmethod
    def from_mapping(cls, raw: Mapping | None) -> "MiniMaxH3StudentSLAConfig":
        raw = {} if raw is None else raw
        enabled = bool(raw.get("enabled", False))
        sparsity = float(raw.get("sparsity", 0.85))
        block_q = int(raw.get("block_q", 64))
        block_k = int(raw.get("block_k", 64))
        dtype_name = str(raw.get("compute_dtype", "bf16")).lower()
        dtypes = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }
        if not 0.0 <= sparsity < 1.0:
            raise ValueError(f"training.dmd.student_sparse_attention.sparsity must satisfy 0 <= sparsity < 1, got {sparsity}.")
        if block_q not in {64, 128}:
            raise ValueError(f"SLA block_q must be 64 or 128, got {block_q}.")
        # The upstream training kernel is validated for 64-wide key blocks.
        if block_k != 64:
            raise ValueError(f"SLA block_k must be 64, got {block_k}.")
        if dtype_name not in dtypes:
            raise ValueError(f"SLA compute_dtype must be bf16 or fp16, got {dtype_name!r}.")
        return cls(
            enabled=enabled,
            sparsity=sparsity,
            block_q=block_q,
            block_k=block_k,
            compute_dtype=dtypes[dtype_name],
        )


class MiniMaxH3SparseOnlyAttnProcessor:
    """Diffusers H3 processor containing SLA's ``o_s`` branch only."""

    def __init__(self, config: MiniMaxH3StudentSLAConfig):
        self.config = config

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise ValueError("MiniMax-H3 student SLA does not support an attention mask. Use padless packed conditions (the H3 DMD cache default).")
        if hidden_states.ndim != 3:
            raise ValueError(f"MiniMax-H3 student SLA expects hidden states [B,L,C], got {tuple(hidden_states.shape)}.")
        if not hidden_states.is_cuda:
            raise RuntimeError("MiniMax-H3 student SLA requires CUDA and Triton.")

        if getattr(attn, "fused_projections", False):
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        input_dtype = query.dtype
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        sparse_map, lut, real_topk = _get_block_map(
            query,
            key,
            keep_ratio=self.config.keep_ratio,
            block_q=self.config.block_q,
            block_k=self.config.block_k,
        )
        routing_key = (
            int(query.shape[-2]),
            self.config.block_q,
            self.config.block_k,
            real_topk,
        )
        if routing_key not in _LOGGED_ROUTING_LAYOUTS:
            _LOGGED_ROUTING_LAYOUTS.add(routing_key)
            total_key_blocks = int(sparse_map.shape[-1])
            logger.info(
                "[train] H3 student SLA routing sequence_length={} key_blocks={} retained_blocks={} actual_sparsity={:.6f}",
                query.shape[-2],
                total_key_blocks,
                real_topk,
                1.0 - real_topk / total_key_blocks,
            )

        # Import lazily so dense H3 training does not require Triton merely by
        # importing its model/trainer modules.
        from .sla_kernel import _attention

        output = _attention.apply(
            query.to(self.config.compute_dtype),
            key.to(self.config.compute_dtype),
            value.to(self.config.compute_dtype),
            sparse_map,
            lut,
            real_topk,
            self.config.block_q,
            self.config.block_k,
        )
        output = output.to(input_dtype).transpose(1, 2).contiguous()
        output = output.flatten(2, 3)
        output = attn.to_out[0](output)
        return attn.to_out[1](output)


def install_minimax_h3_student_sla(transformer, config: MiniMaxH3StudentSLAConfig) -> int:
    """Replace only the student's main DiT block processors with sparse SLA."""
    if not config.enabled:
        return 0
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        raise TypeError("MiniMax-H3 transformer has no transformer_blocks to patch with SLA.")
    if not torch.cuda.is_available():
        raise RuntimeError("MiniMax-H3 student SLA requires a CUDA runtime.")
    try:
        from . import sla_kernel as _sla_kernel  # noqa: F401
    except ImportError as error:
        raise ImportError("MiniMax-H3 student SLA requires Triton. Install a Triton build compatible with PyTorch.") from error

    count = 0
    for block in blocks:
        attention = getattr(block, "attn", None)
        if attention is None or not hasattr(attention, "set_processor"):
            raise TypeError("MiniMax-H3 transformer block attention does not expose set_processor().")
        if int(getattr(attention, "head_dim", -1)) not in {64, 128}:
            raise ValueError(f"The bundled SLA Triton kernel requires attention head_dim 64 or 128, got {getattr(attention, 'head_dim', None)}.")
        attention.set_processor(MiniMaxH3SparseOnlyAttnProcessor(config))
        count += 1
    if count == 0:
        raise RuntimeError("MiniMax-H3 student SLA did not replace any attention processors.")

    logger.info(
        "[train] installed student-only H3 SLA sparse attention: blocks={} requested_sparsity={:.4f} keep_ratio={:.4f} block_q={} block_k={} compute_dtype={} linear_branch=false text_refiner=dense",
        count,
        config.sparsity,
        config.keep_ratio,
        config.block_q,
        config.block_k,
        str(config.compute_dtype).removeprefix("torch."),
    )
    return count


__all__ = [
    "MiniMaxH3SparseOnlyAttnProcessor",
    "MiniMaxH3StudentSLAConfig",
    "install_minimax_h3_student_sla",
    "retained_key_blocks",
]
