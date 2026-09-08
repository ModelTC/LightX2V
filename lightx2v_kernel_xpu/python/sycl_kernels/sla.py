"""SLA routing and fused block-sparse attention for Intel XPU.

The public layout is BLHD, matching :func:`sycl_kernels.cute_sdp` and the
MiniMax-H3 attention path.  Routing is intentionally expressed with PyTorch
XPU operations; the optimized QK/softmax/PV path is one CUTE kernel and does
not materialize the sparse score matrix. Triton is retained as a fallback.
"""

from __future__ import annotations

import math

import torch


def _block_mean_blhd(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Return block means as [B, H, ceil(L / block_size), D]."""
    if x.ndim != 4:
        raise ValueError(f"expected a BLHD tensor, got shape {tuple(x.shape)}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    batch, length, heads, dim = x.shape
    full_blocks, tail = divmod(length, block_size)
    pieces = []
    if full_blocks:
        prefix = x[:, : full_blocks * block_size]
        pieces.append(prefix.reshape(batch, full_blocks, block_size, heads, dim).mean(dim=2))
    if tail:
        pieces.append(x[:, full_blocks * block_size :].mean(dim=1, keepdim=True))
    if not pieces:
        raise ValueError("sequence length must be non-zero")
    # [B, blocks, H, D] -> [B, H, blocks, D].  The small pooled tensor is
    # made contiguous because it is consumed by batched matmul immediately.
    return torch.cat(pieces, dim=1).permute(0, 2, 1, 3).contiguous()


def sla_block_map(
    q: torch.Tensor,
    k: torch.Tensor,
    keep_ratio: float = 0.2,
    block_q: int = 128,
    block_k: int = 128,
) -> torch.Tensor:
    """Build the SLA key-block LUT for BLHD Q and K tensors.

    Returns int32 ``[B, Hq, ceil(Lq/block_q), topk]`` indices.  Smooth-K is
    applied in pooled form: ``mean(block(K)) - mean(sequence(K))``.  This is
    algebraically identical to centering the full K tensor first, while
    avoiding an additional full-sequence allocation.
    """
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must use [B, L, H, D] layout")
    if q.device != k.device or q.dtype != k.dtype:
        raise ValueError("q and k must share device and dtype")
    if q.shape[0] != k.shape[0] or q.shape[3] != k.shape[3]:
        raise ValueError("q and k batch/head_dim must match")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    hq, hkv = q.shape[2], k.shape[2]
    if hq % hkv != 0:
        raise ValueError(f"query heads ({hq}) must be divisible by KV heads ({hkv})")

    pooled_q = _block_mean_blhd(q, block_q)
    pooled_k = _block_mean_blhd(k, block_k)
    pooled_k = pooled_k - k.mean(dim=1).unsqueeze(2)
    if hq != hkv:
        pooled_k = pooled_k.repeat_interleave(hq // hkv, dim=1)

    scores = torch.matmul(pooled_q, pooled_k.transpose(-1, -2))
    key_blocks = scores.shape[-1]
    topk = max(1, min(key_blocks, int(keep_ratio * key_blocks)))
    # Sorting is unnecessary for online softmax and costs a measurable amount
    # at long sequence lengths.  int32 halves LUT traffic in the fused kernel.
    return torch.topk(scores, topk, dim=-1, sorted=False).indices.to(torch.int32).contiguous()


def sparse_block_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lut: torch.Tensor,
    block_q: int = 128,
    block_k: int = 128,
    scale: float | None = None,
) -> torch.Tensor:
    """Slow, device-independent reference for tests and bring-up."""
    _validate_sparse_inputs(q, k, v, lut, block_q, block_k)
    batch, q_len, q_heads, dim = q.shape
    kv_len, kv_heads = k.shape[1], k.shape[2]
    group_size = q_heads // kv_heads
    scale = dim**-0.5 if scale is None else scale
    out = torch.empty_like(q)
    offsets = torch.arange(block_k, device=lut.device, dtype=torch.long)

    for b in range(batch):
        for h in range(q_heads):
            kv_h = h // group_size
            for qb in range(lut.shape[2]):
                q_start = qb * block_q
                q_stop = min(q_start + block_q, q_len)
                block_ids = lut[b, h, qb].long()
                positions = (block_ids[:, None] * block_k + offsets[None, :]).reshape(-1)
                positions = positions[positions < kv_len]
                q_tile = q[b, q_start:q_stop, h].float()
                k_tile = k[b, positions, kv_h].float()
                v_tile = v[b, positions, kv_h].float()
                probs = torch.softmax(torch.matmul(q_tile, k_tile.transpose(0, 1)) * scale, dim=-1)
                out[b, q_start:q_stop, h] = torch.matmul(probs, v_tile).to(out.dtype)
    return out


def _validate_sparse_inputs(q, k, v, lut, block_q: int, block_k: int) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must use [B, L, H, D] layout")
    if not (q.device == k.device == v.device == lut.device):
        raise ValueError("q, k, v and lut must be on the same device")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError("q, k and v must share dtype")
    if q.shape[0] != k.shape[0] or k.shape != v.shape:
        raise ValueError("batch must match and k/v shapes must be identical")
    if q.shape[3] != k.shape[3]:
        raise ValueError("q and k/v head_dim must match")
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError("query head count must be divisible by KV head count")
    expected_q_blocks = math.ceil(q.shape[1] / block_q)
    if lut.ndim != 4 or tuple(lut.shape[:3]) != (q.shape[0], q.shape[2], expected_q_blocks):
        raise ValueError(
            "lut must have shape [B, Hq, ceil(Lq/block_q), topk], got "
            f"{tuple(lut.shape)}"
        )
    if lut.shape[3] == 0:
        raise ValueError("lut topk dimension must be non-zero")
    if block_q <= 0 or block_k <= 0:
        raise ValueError("block sizes must be positive")


def sparse_block_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lut: torch.Tensor,
    block_q: int = 128,
    block_k: int = 128,
    scale: float | None = None,
) -> torch.Tensor:
    """Run fused block-sparse attention on Intel XPU.

    QK, online softmax, and PV are fused.  No score tensor is written to
    global memory.  The first optimized contract is BF16, D=128, forward-only.
    """
    _validate_sparse_inputs(q, k, v, lut, block_q, block_k)
    if q.device.type != "xpu":
        raise ValueError("sparse_block_attention requires Intel XPU tensors")
    if q.dtype != torch.bfloat16 or q.shape[-1] != 128:
        raise ValueError("the optimized XPU path currently requires BF16 and head_dim=128")
    if block_q not in (64, 128) or block_k not in (64, 128):
        raise ValueError("the optimized XPU path supports block sizes 64 or 128")

    if block_q == 128 and block_k == 128 and scale is None and q.shape[2] == k.shape[2]:
        # The CUTE path preserves the tuned dense kernel's 2D block loads,
        # DPAS pipeline, and online softmax; the LUT only redirects K/V tiles.
        from . import _load_cute_fmha_minimax_h3_sparse

        try:
            op = torch.ops.sycl_kernels_cute_minimax_h3_sparse.sparse_sdp
        except AttributeError:
            _load_cute_fmha_minimax_h3_sparse()
            op = torch.ops.sycl_kernels_cute_minimax_h3_sparse.sparse_sdp
        return op(q, k, v, lut.to(torch.int32).contiguous())

    # Triton remains as a correctness-oriented fallback for GQA, custom scale,
    # and 64-sized blocks while their specialized CUTE variants are developed.
    from .sla_triton import launch_sparse_block_attention

    return launch_sparse_block_attention(
        q.contiguous(), k.contiguous(), v.contiguous(), lut.to(torch.int32).contiguous(),
        block_q, block_k, q.shape[-1] ** -0.5 if scale is None else scale,
    )


def sla_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep_ratio: float = 0.2,
    block_q: int = 128,
    block_k: int = 128,
    scale: float | None = None,
) -> torch.Tensor:
    """Convenience entry point combining SLA routing and sparse attention."""
    lut = sla_block_map(q, k, keep_ratio, block_q, block_k)
    return sparse_block_attention(q, k, v, lut, block_q, block_k, scale)
