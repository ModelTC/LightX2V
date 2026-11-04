"""Sparse attention contracts: fixed top-k, actual masks, and actual ranges.

QK+PV FLOPs per Q/K token pair = 2 * (QK dimension + V dimension).
Logical pairs exclude tail padding; scheduled pairs include full tiles. This module
normalizes inputs and computes areas without session/model/profiler state; the base exposes these helpers to backends.
"""

from lightx2v.common.ops.compute_stats import attention_flops, range_token_pairs, selected_token_pairs


def _geometry(q, k, v, block_q, block_k, layout):
    """Normalize shapes to BSHD without converting or reading Q/K/V values."""
    shapes = [tuple(tensor.shape) for tensor in (q, k, v)]
    ndim = 3 if layout == "SHD" else 4
    if layout not in {"SHD", "BHSD", "BSHD"} or any(len(shape) != ndim for shape in shapes):
        raise ValueError("sparse compute contract requires Q/K/V matching explicit SHD, BHSD or BSHD layout")
    if type(block_q) is not int or type(block_k) is not int or min(block_q, block_k) < 1:
        raise ValueError("sparse compute contract requires positive integer block sizes")
    if layout == "SHD":
        shapes = [(1, *shape) for shape in shapes]
    elif layout == "BHSD":
        shapes = [(b, s, h, d) for b, h, s, d in shapes]
    (batch, sq, heads, dim), (k_batch, sk, kv_heads, dk), (v_batch, sv, v_heads, dv) = shapes
    if min(batch, sq, sk, heads, kv_heads, dim, dv) < 1 or batch != k_batch or batch != v_batch or sk != sv or kv_heads != v_heads or dim != dk or heads % kv_heads:
        raise ValueError("sparse compute contract requires compatible nonempty Q/K/V shapes")
    shape = {"batch": batch, "seq_q": sq, "seq_kv": sk, "heads": heads, "kv_heads": kv_heads, "head_dim": dim, "causal": False}
    return shape, attention_flops(1, dim, dv)


def _work(shape, factor, selected_pairs, scheduled_pairs, definition):
    """Share one area-to-FLOPs conversion across all three contracts and backends."""
    dense_pairs = shape["batch"] * shape["heads"] * shape["seq_q"] * shape["seq_kv"]
    return {
        "dense_equivalent_flops": factor * dense_pairs,
        "selected_logical_flops": factor * selected_pairs if selected_pairs is not None else None,
        "scheduled_tile_flops": factor * scheduled_pairs if scheduled_pairs is not None else None,
        "definition": definition,
    }


def from_sparsity(q, k, v, sparsity, block_q=1, block_k=1, *, min_blocks=0, layout="BHSD"):
    """Select floor((1-sparsity)*K_blocks) blocks per Q block using shapes only."""
    from math import ceil
    from numbers import Real

    shape, factor = _geometry(q, k, v, block_q, block_k, layout)
    if isinstance(sparsity, bool) or not isinstance(sparsity, Real) or not 0 <= sparsity <= 1:
        raise ValueError("sparsity must be a finite number in [0, 1], meaning the discarded K-block fraction")
    if type(min_blocks) is not int or min_blocks < 0:
        raise ValueError("min_blocks must be a nonnegative integer")
    sparsity = float(sparsity)
    batch, heads, sq, sk = (shape[key] for key in ("batch", "heads", "seq_q", "seq_kv"))
    nq, nk = ceil(sq / block_q), ceil(sk / block_k)
    topk = min(nk, max(min_blocks, int((1 - sparsity) * nk)))
    # Empty/full selection determines tail coverage; partial selection with a K tail does not.
    if topk == 0:
        pairs = 0
    elif topk == nk:
        pairs = sq * sk
    elif sk % block_k == 0:
        pairs = sq * topk * block_k
    else:
        pairs = None
    return {
        "shape": shape,
        "sparse": {"sparsity": sparsity, "block_q": block_q, "block_k": block_k, "tiles_per_query_block": topk},
        "work": _work(
            shape,
            factor,
            batch * heads * pairs if pairs is not None else None,
            batch * heads * nq * topk * block_q * block_k,
            "Shape-derived fixed-topk QK+PV; scheduled includes padding; partial selection with a K tail requires the actual mask; excludes routing/softmax/quantization",
        ),
    }


def from_mask(q, k, v, sparse_map, block_q=1, block_k=1, *, layout="BHSD", q_block_sizes=None, k_block_sizes=None):
    """Count QK/PV from a per-query-head mask; trim fixed-block tails or use actual BHQ/BHK variable sizes."""
    from math import ceil

    import torch

    shape, factor = _geometry(q, k, v, block_q, block_k, layout)
    batch, heads, sq, sk = (shape[key] for key in ("batch", "heads", "seq_q", "seq_kv"))
    variable_blocks = q_block_sizes is not None or k_block_sizes is not None
    if variable_blocks:
        if q_block_sizes is None or k_block_sizes is None or block_q != 1 or block_k != 1:
            raise ValueError("variable blocks require both size arrays and no fixed block size")
        q_sizes, k_sizes = (torch.as_tensor(sizes).detach().cpu() for sizes in (q_block_sizes, k_block_sizes))
        if any(sizes.ndim != 3 or tuple(sizes.shape[:2]) != (batch, heads) or sizes.is_floating_point() or sizes.is_complex() or sizes.dtype == torch.bool for sizes in (q_sizes, k_sizes)):
            raise ValueError("variable block sizes must be integer BHQ/BHK arrays")
        if bool((q_sizes < 0).any()) or bool((k_sizes < 0).any()) or not bool((q_sizes.sum(-1) == sq).all()) or not bool((k_sizes.sum(-1) == sk).all()):
            raise ValueError("variable block sizes must be nonnegative partitions of Q/K sequences")
        q_sizes, k_sizes = q_sizes.to(torch.int64), k_sizes.to(torch.int64)
        expected_mask = (batch, heads, q_sizes.shape[-1], k_sizes.shape[-1])
    else:
        expected_mask = (batch, heads, ceil(sq / block_q), ceil(sk / block_k))
        q_sizes = (sq - torch.arange(expected_mask[-2]) * block_q).clamp(max=block_q)
        k_sizes = (sk - torch.arange(expected_mask[-1]) * block_k).clamp(max=block_k)
    if tuple(sparse_map.shape) != expected_mask:
        raise ValueError("sparse compute contract requires mask tiles matching actual Q/K/V")
    # Read the sparse_map consumed by the production kernel; CPU copies/reductions are diagnostic-only.
    mask = sparse_map.detach().cpu()
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("sparse map must contain only selected/unselected binary tiles")
    selected = int(mask.sum().item())
    routing = {"source": "production_sparse_map", "shape": list(mask.shape), "selected_tiles": selected, "total_tiles": mask.numel(), "block_q": block_q, "block_k": block_k}
    # Use the same area formula for fixed blocks, tails, and variable blocks.
    pairs = selected_token_pairs(mask, q_sizes, k_sizes)
    scheduled_pairs = None if variable_blocks else selected * block_q * block_k
    if variable_blocks:
        routing.update(block_q=None, block_k=None, q_block_sizes=q_sizes.tolist(), k_block_sizes=k_sizes.tolist())
    return {
        "operator_family": "sparse_attention",
        "shape": shape,
        "routing": routing,
        "work": {
            **_work(
                shape,
                factor,
                pairs,
                scheduled_pairs,
                "QK+PV from actual variable block sizes; internal kernel padding unknown; excludes routing, quantization and softmax; not hardware instruction counts"
                if variable_blocks
                else "QK+PV only; scheduled includes tile padding; excludes routing, quantization and softmax; not hardware instruction counts",
            ),
            "physical_instructions": None,
        },
    }


def from_ranges(q, k, v, q_ranges, k_ranges, *, layout="SHD", range_repeat=1):
    """Count non-overlapping ranges; packed heads are included, while Radial explicitly repeats ranges per head."""
    import torch

    shape, factor = _geometry(q, k, v, 1, 1, layout)
    if shape["heads"] != shape["kv_heads"]:
        raise ValueError("sparse range contract currently requires equal Q/KV heads")
    ranges = [tensor.detach().cpu() for tensor in (q_ranges, k_ranges)]
    if (
        type(range_repeat) is not int
        or range_repeat < 1
        or any(tensor.ndim != 2 or tensor.shape[-1] != 2 or tensor.is_floating_point() or tensor.is_complex() or tensor.dtype == torch.bool for tensor in ranges)
        or ranges[0].shape != ranges[1].shape
    ):
        raise ValueError("ranges require matched integer [N,2] tensors and positive repeat")
    if any(bool((tensor[:, 1] < tensor[:, 0]).any()) or bool((tensor < 0).any()) for tensor in ranges):
        raise ValueError("ranges require nonnegative ordered endpoints")
    heads = shape["batch"] * shape["heads"]
    if heads % range_repeat or any(bool((tensor[:, 1] > heads // range_repeat * length).any()) for tensor, length in zip(ranges, (shape["seq_q"], shape["seq_kv"]))):
        raise ValueError("ranges must stay within the actual local head/sequence coordinates")
    pairs = range_token_pairs(*ranges) * range_repeat
    return {
        "operator_family": "sparse_attention",
        "shape": shape,
        "routing": {"source": "production_nonoverlapping_ranges", "rectangles": ranges[0].shape[0], "range_repeat": range_repeat},
        "work": _work(shape, factor, pairs, None, "QK+PV from nonoverlapping production rectangles; excludes routing/softmax/quantization; internal tile padding unknown"),
    }
