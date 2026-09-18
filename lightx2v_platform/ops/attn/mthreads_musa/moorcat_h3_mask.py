# SPDX-License-Identifier: Apache-2.0
"""Cube layout and top-k block selection for MiniMax-H3."""

import math
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class PackedStreams:
    """Packed token indices and per-sample condition metadata."""

    text: torch.Tensor
    cond_image: torch.Tensor
    latent: torch.Tensor
    cond_audio: torch.Tensor
    audio: torch.Tensor
    cond_image_shapes: list
    cond_image_roles: list
    cond_event_orders: list
    cond_audio_stream_lens: list

    def as_index_tuple(self):
        return (self.text, self.cond_image, self.latent, self.cond_audio, self.audio)


@dataclass(frozen=True)
class CubeLayout:
    """Request-static metadata in label, physical-block, and token spaces."""

    # ── semantic-label space (leading dim L, unless noted) ───────────────
    topk_mask: torch.Tensor
    base_block_mask: torch.Tensor
    sparse_label_mask: torch.Tensor
    label_lengths: torch.Tensor
    label_block_counts: torch.Tensor
    label_block_indices: torch.Tensor
    max_label_block_count: int
    topk_semantic_capacity: int

    # ── physical-block space (leading dim B) ─────────────────────────────
    num_blocks: int
    block_labels: torch.Tensor
    base_block_counts: torch.Tensor
    base_block_indices: torch.Tensor

    # ── token space (packed and cube-padded) ────────────────────────────
    real_total_len: int
    padded_seqlen: int
    is_real: torch.Tensor
    pad_indices: torch.Tensor
    sorted_real_indices: torch.Tensor
    sorted_real_indices_i32: torch.Tensor
    label_offsets_i32: torch.Tensor
    expand_indices: torch.Tensor
    gather_indices: torch.Tensor


@dataclass
class CubeRuntime:
    """Buffers reused across transformer blocks within one request."""

    kv_num_blocks_buffer: torch.Tensor | None = None
    kv_indices_buffer: torch.Tensor | None = None
    token_labels: torch.Tensor | None = None
    sparse_base_capacity: int = 0


@dataclass
class CubePrecomputed:
    layout: CubeLayout
    runtime: CubeRuntime = field(default_factory=CubeRuntime)


def normalize_condition_event_order(events, visual_count, audio_count):
    events = tuple(events)
    event_types = {event_type for event_type, _ in events}
    unsupported = event_types - {"imgvid", "audio"}
    if unsupported:
        raise ValueError(f"unsupported condition event types: {sorted(unsupported)}")
    imgvid_indices = [index for item_type, index in events if item_type == "imgvid"]
    if imgvid_indices != list(range(visual_count)):
        raise ValueError(f"condition imgvid indices {imgvid_indices} do not cover {visual_count} tensors")
    audio_indices = [index for item_type, index in events if item_type == "audio"]
    if audio_indices != list(range(audio_count)):
        raise ValueError(f"condition audio indices {audio_indices} do not cover {audio_count} tensors")
    return events


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _stable_sort_1d(values):
    """Sort metadata on CPU because muDNN cannot stably sort these vectors."""
    if values.device.type != "cpu":
        sorted_values, order = torch.sort(values.cpu(), stable=True)
        return sorted_values.to(values.device), order.to(values.device)
    return torch.sort(values, stable=True)


def _cube_token_size(cube_size):
    cube_size = tuple(int(value) for value in cube_size)
    if len(cube_size) != 3 or any(value <= 0 for value in cube_size):
        raise ValueError(f"local_cube_size must be a positive 3D size, got {cube_size}")
    token_size = math.prod(cube_size)
    if token_size & (token_size - 1):
        raise ValueError(f"local_cube_size product must be a power of two for Moorcat, got {cube_size} ({token_size} tokens)")
    return token_size


def _normalize_cube_visual_shape(shape):
    shape = tuple(int(value) for value in shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"cube attention visual shapes must be positive 3D shapes, got {shape}")
    return shape


_COND_IMAGE_ROLES = frozenset({"joint_cube", "independent_cube", "dense_prefix"})


def _normalize_cond_image_roles(roles, visual_count, sample_idx):
    roles = tuple(roles)
    if len(roles) != visual_count:
        raise ValueError(f"cube attention sample {sample_idx} has {visual_count} condition visual streams but {len(roles)} condition roles")
    unsupported = sorted(set(roles) - _COND_IMAGE_ROLES)
    if unsupported:
        raise ValueError(f"cube attention sample {sample_idx} has unsupported condition visual roles: {unsupported}")
    return roles


def _cube_sample_segments(indices, cu_seqlens):
    bounds = torch.searchsorted(indices, cu_seqlens).tolist()
    return [indices[start:end] for start, end in zip(bounds[:-1], bounds[1:])]


def _group_cube_visual_segment(indices, shape, cube_size, sample_idx, segment_name):
    expected = math.prod(shape)
    if indices.numel() != expected:
        raise ValueError(f"cube attention {segment_name} for sample {sample_idx} has {indices.numel()} tokens, expected {expected} for shape {shape}")
    cube_counts = tuple(_ceil_div(dim, extent) for dim, extent in zip(shape, cube_size))
    linear_indices = torch.arange(
        expected,
        dtype=torch.long,
        device=indices.device,
    )
    block_labels = torch.zeros_like(linear_indices)
    for coordinate, extent, count in zip(
        torch.unravel_index(linear_indices, shape),
        cube_size,
        cube_counts,
    ):
        block_labels = block_labels * count + coordinate // extent
    ordered_labels, order = _stable_sort_1d(block_labels)
    return indices.index_select(0, order), ordered_labels, math.prod(cube_counts)


def _rank_position_axis(values):
    """Map one floating position axis to stable, zero-based unique ranks."""
    _, order = _stable_sort_1d(values)
    sorted_values = values.index_select(0, order)
    changed = torch.empty_like(sorted_values, dtype=torch.bool)
    changed[0] = True
    changed[1:] = sorted_values[1:] != sorted_values[:-1]
    sorted_ranks = changed.to(torch.long).cumsum(0) - 1
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def _group_joint_cube_visual_segment(
    indices,
    position_ids,
    cube_size,
):
    """Use ranked RoPE coordinates to share labels with embedded keyframes."""
    positions = position_ids.index_select(0, indices)
    ranked_axes = tuple(_rank_position_axis(positions[:, axis]) for axis in range(3))
    cube_counts = tuple(int(axis.max().item()) // extent + 1 for axis, extent in zip(ranked_axes, cube_size))
    block_labels = torch.zeros(indices.numel(), dtype=torch.long, device=indices.device)
    for coordinate, extent, count in zip(ranked_axes, cube_size, cube_counts):
        block_labels = block_labels * count + coordinate // extent
    ordered_labels, order = _stable_sort_1d(block_labels)
    return indices.index_select(0, order), ordered_labels, math.prod(cube_counts)


def _group_cube_1d_segment(indices, cube_token_size):
    local_labels = (
        torch.arange(
            indices.numel(),
            dtype=torch.long,
            device=indices.device,
        )
        // cube_token_size
    )
    return indices, local_labels, _ceil_div(indices.numel(), cube_token_size)


def _validate_cube_sample_metadata(values, num_samples, name):
    if len(values) != num_samples:
        raise ValueError(f"cube attention received {len(values)} {name} entries for {num_samples} samples")
    return values


def _split_cube_streams(indices, stream_sizes, sample_idx, stream_name):
    streams = []
    offset = 0
    for stream_size in stream_sizes:
        stream_size = int(stream_size)
        if stream_size < 0:
            raise ValueError(f"cube attention {stream_name} stream sizes must be nonnegative")
        streams.append(indices[offset : offset + stream_size])
        offset += stream_size
    if offset != indices.numel():
        raise ValueError(f"cube attention {stream_name} streams for sample {sample_idx} cover {offset} tokens, but packing contains {indices.numel()} tokens")
    return streams


def _pack_block_rows(block_mask):
    """Pack a head-independent ``[Q, KV]`` mask without sorting."""
    counts = block_mask.sum(dim=-1, dtype=torch.int32)
    capacity = int(counts.max().item()) if counts.numel() else 0
    indices = torch.zeros(block_mask.shape[0], capacity, dtype=torch.int32, device=block_mask.device)
    if capacity:
        positions = block_mask.to(torch.int32).cumsum(dim=-1) - 1
        row_ids, column_ids = block_mask.nonzero(as_tuple=True)
        indices[row_ids, positions[row_ids, column_ids].to(torch.long)] = column_ids.to(torch.int32)
    return counts[None, None], indices[None, None]


def _build_physical_base_layout(base_block_mask, block_labels):
    physical = base_block_mask[block_labels[:, None], block_labels[None, :]]
    return _pack_block_rows(physical)


def _raise_for_unoccupied_labels(occupied_labels, sample_label_ranges):
    """Reject holes introduced by a condition stream on a different grid."""
    occupied = set(occupied_labels.tolist())
    for sample_idx, (start, end) in enumerate(sample_label_ranges):
        missing = [label for label in range(start, end) if label not in occupied]
        if missing:
            raise ValueError(
                f"cube attention sample {sample_idx} allocated labels "
                f"[{start}, {end}) but {len(missing)} of them received no "
                f"tokens (first missing: {missing[0]}). A joint_cube "
                "condition visual stream does not share the target's "
                "position grid; declare it independent_cube or dense_prefix "
                "instead."
            )
    raise ValueError(f"cube attention allocated labels outside every sample range; occupied {len(occupied)} labels for ranges {sample_label_ranges}")


def _build_cube_segment_layout(
    sample_shapes,
    cu_seqlens,
    real_total_len,
    cube_size,
    device,
    streams,
    position_ids,
):
    cube_token_size = math.prod(cube_size)
    num_samples = len(sample_shapes)
    if cu_seqlens.numel() != num_samples + 1:
        raise ValueError(f"cube attention received {cu_seqlens.numel() - 1} packed sequences for {num_samples} target shapes")
    cond_image_shapes = _validate_cube_sample_metadata(
        streams.cond_image_shapes,
        num_samples,
        "condition-shape",
    )
    cond_image_roles = _validate_cube_sample_metadata(
        streams.cond_image_roles,
        num_samples,
        "condition-role",
    )
    cond_event_orders = _validate_cube_sample_metadata(
        streams.cond_event_orders,
        num_samples,
        "condition-event",
    )
    cond_audio_stream_lens = _validate_cube_sample_metadata(
        streams.cond_audio_stream_lens,
        num_samples,
        "condition-audio",
    )
    (
        text_segments,
        cond_image_segments,
        latent_segments,
        cond_audio_segments,
        audio_segments,
    ) = [
        _cube_sample_segments(
            indices.to(device=device, dtype=torch.long),
            cu_seqlens,
        )
        for indices in streams.as_index_tuple()
    ]

    cube_labels = torch.full(
        (real_total_len,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    label_offset = 0
    sample_label_ranges = []
    sparse_labels = []
    ordered_segments = []

    def add_segment(ordered_indices, local_block_labels, num_blocks, *, sparse):
        nonlocal label_offset
        if ordered_indices.numel() == 0:
            return
        assigned = cube_labels.index_select(0, ordered_indices) >= 0
        if assigned.any():
            duplicate = int(ordered_indices[torch.nonzero(assigned)[0, 0]])
            raise ValueError(f"packed token {duplicate} belongs to multiple streams")
        cube_labels[ordered_indices] = (local_block_labels + label_offset).to(torch.int32)
        ordered_segments.append(ordered_indices)
        sparse_labels.extend([sparse] * num_blocks)
        label_offset += num_blocks

    for sample_idx in range(num_samples):
        text = text_segments[sample_idx]
        cond_image = cond_image_segments[sample_idx]
        latent = latent_segments[sample_idx]
        cond_audio = cond_audio_segments[sample_idx]
        target_audio = audio_segments[sample_idx]
        sample_label_start = label_offset

        cond_audio_streams = _split_cube_streams(
            cond_audio,
            cond_audio_stream_lens[sample_idx],
            sample_idx,
            "condition audio",
        )
        visual_shapes = [_normalize_cube_visual_shape(shape) for shape in cond_image_shapes[sample_idx]]
        raw_visual_streams = _split_cube_streams(
            cond_image,
            [math.prod(shape) for shape in visual_shapes],
            sample_idx,
            "condition visual",
        )
        visual_roles = _normalize_cond_image_roles(cond_image_roles[sample_idx], len(visual_shapes), sample_idx)

        events = normalize_condition_event_order(
            cond_event_orders[sample_idx],
            visual_count=len(raw_visual_streams),
            audio_count=len(cond_audio_streams),
        )
        add_segment(
            *_group_cube_1d_segment(text, cube_token_size),
            sparse=False,
        )
        for event_type, event_idx in events:
            if event_type == "audio":
                add_segment(
                    *_group_cube_1d_segment(
                        cond_audio_streams[event_idx],
                        cube_token_size,
                    ),
                    sparse=False,
                )
            else:
                shape = visual_shapes[event_idx]
                role = visual_roles[event_idx]
                if role == "joint_cube":
                    continue
                if role == "independent_cube" and shape[0] <= 1:
                    raise ValueError(f"independent_cube condition visual streams must have a genuine 3D shape, got {shape}")
                if role == "independent_cube":
                    grouped = _group_cube_visual_segment(
                        raw_visual_streams[event_idx],
                        shape,
                        cube_size,
                        sample_idx,
                        f"condition visual stream {event_idx}",
                    )
                else:
                    grouped = _group_cube_1d_segment(raw_visual_streams[event_idx], cube_token_size)
                add_segment(*grouped, sparse=role == "independent_cube")
        add_segment(
            *_group_cube_1d_segment(target_audio, cube_token_size),
            sparse=False,
        )

        target_shape = sample_shapes[sample_idx]
        expected_target_tokens = math.prod(target_shape)
        if latent.numel() != expected_target_tokens:
            raise ValueError(f"cube attention target visual for sample {sample_idx} has {latent.numel()} tokens, expected {expected_target_tokens} for shape {target_shape}")
        joint_streams = [stream for stream, role in zip(raw_visual_streams, visual_roles) if role == "joint_cube"]
        joint_indices = torch.cat([*joint_streams, latent])
        add_segment(
            *_group_joint_cube_visual_segment(
                joint_indices,
                position_ids,
                cube_size,
            ),
            sparse=True,
        )
        sample_label_ranges.append((sample_label_start, label_offset))

    unassigned = torch.nonzero(cube_labels < 0, as_tuple=False).flatten()
    if unassigned.numel():
        first_token = int(unassigned[0])
        first_sample = int(torch.searchsorted(cu_seqlens, first_token, right=True)) - 1
        raise ValueError(f"packed token {first_token} in sample {first_sample} is not assigned to a stream")
    sort_idx = torch.cat(ordered_segments)

    return (
        cube_labels,
        sort_idx,
        sample_label_ranges,
        torch.tensor(sparse_labels, dtype=torch.bool, device=device),
    )


def precompute_cube_attention(
    sample_shapes,
    cu_seqlens,
    total_len,
    cube_size,
    device,
    streams,
    position_ids,
    max_sparse_topk_ratio,
):
    """Build request-static cube metadata for a packed batch."""
    sample_shapes = [_normalize_cube_visual_shape(shape) for shape in sample_shapes]
    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long)
    real_total_len = int(cu_seqlens[-1].item())
    if real_total_len > total_len:
        raise ValueError(f"cube attention real length {real_total_len} exceeds total length {total_len}")
    cube_token_size = _cube_token_size(cube_size)
    cube_size = tuple(int(value) for value in cube_size)
    position_ids = position_ids.to(device=device)
    if position_ids.ndim != 2 or position_ids.shape != (total_len, 3):
        raise ValueError(f"cube attention img_position_ids must have shape [{total_len}, 3], got {tuple(position_ids.shape)}")
    cube_labels, sort_idx, sample_label_ranges, sparse_label_mask = _build_cube_segment_layout(
        sample_shapes,
        cu_seqlens,
        real_total_len,
        cube_size,
        device,
        streams,
        position_ids,
    )
    num_labels = sum(end - start for start, end in sample_label_ranges)
    dead_label = num_labels

    topk_mask = torch.zeros(num_labels, num_labels, dtype=torch.bool, device=device)
    base_block_mask = torch.zeros(num_labels, num_labels, dtype=torch.bool, device=device)
    base_block_mask.fill_diagonal_(True)
    for start, end in sample_label_ranges:
        sample_sparse = sparse_label_mask[start:end]
        topk_mask[start:end, start:end] = sample_sparse.unsqueeze(1) & sample_sparse.unsqueeze(0)
        sample_dense = ~sample_sparse
        base_block_mask[start:end, start:end] |= sample_dense.unsqueeze(1) | sample_dense.unsqueeze(0)

    sorted_labels = cube_labels[sort_idx]
    occupied_labels, counts_per_label = sorted_labels.unique_consecutive(return_counts=True)
    if occupied_labels.numel() != num_labels:
        _raise_for_unoccupied_labels(occupied_labels, sample_label_ranges)
    padded_counts = ((counts_per_label + cube_token_size - 1) // cube_token_size) * cube_token_size

    padded_offsets = torch.zeros(len(padded_counts) + 1, dtype=torch.long, device=device)
    padded_offsets[1:] = padded_counts.cumsum(0)
    group_starts = torch.zeros(len(counts_per_label) + 1, dtype=torch.long, device=device)
    group_starts[1:] = counts_per_label.cumsum(0)

    sorted_positions = torch.arange(real_total_len, device=device)
    group_idx = torch.bucketize(sorted_positions, group_starts[1:], right=True)
    padded_pos_sorted = padded_offsets[group_idx] + (sorted_positions - group_starts[group_idx])

    expand_indices = torch.empty(real_total_len, dtype=torch.long, device=device)
    expand_indices[sort_idx] = padded_pos_sorted

    padded_seqlen = int(padded_offsets[-1].item())
    label_block_counts = padded_counts // cube_token_size
    label_block_offsets = torch.zeros(num_labels + 1, dtype=torch.long, device=device)
    label_block_offsets[1:] = label_block_counts.cumsum(0)
    num_blocks = int(label_block_offsets[-1].item())
    block_labels = torch.repeat_interleave(
        torch.arange(num_labels, dtype=torch.long, device=device),
        label_block_counts,
    )
    max_label_block_count = int(label_block_counts.max().item())
    label_block_slots = torch.arange(max_label_block_count, dtype=torch.long, device=device)
    label_block_indices = label_block_offsets[:-1, None] + label_block_slots
    label_block_indices = torch.where(
        label_block_slots < label_block_counts[:, None],
        label_block_indices,
        torch.zeros_like(label_block_indices),
    )
    gather_indices = torch.zeros(padded_seqlen, dtype=torch.long, device=device)
    gather_indices[padded_pos_sorted] = sort_idx

    padded_cube_labels = torch.full((padded_seqlen,), dead_label, dtype=torch.int32, device=device)
    padded_cube_labels[padded_pos_sorted] = sorted_labels

    is_real = (padded_cube_labels != dead_label).to(torch.int32)
    pad_indices = torch.nonzero(is_real == 0, as_tuple=False).squeeze(1)
    label_lengths = counts_per_label.to(torch.long)
    base_block_counts, base_block_indices = _build_physical_base_layout(base_block_mask, block_labels)
    sparse_sizes = topk_mask.sum(dim=-1)
    if max_sparse_topk_ratio > 0:
        max_selected_counts = (sparse_sizes.to(torch.float32) * float(max_sparse_topk_ratio)).to(torch.long)
        max_selected_counts.clamp_(min=1)
        max_selected_counts = torch.minimum(max_selected_counts, sparse_sizes)
        topk_semantic_capacity = int(max_selected_counts.max().item())
    else:
        topk_semantic_capacity = 0

    return CubePrecomputed(
        layout=CubeLayout(
            topk_mask=topk_mask,
            base_block_mask=base_block_mask,
            sparse_label_mask=sparse_label_mask,
            label_lengths=label_lengths,
            label_block_counts=label_block_counts,
            label_block_indices=label_block_indices,
            max_label_block_count=max_label_block_count,
            topk_semantic_capacity=topk_semantic_capacity,
            num_blocks=num_blocks,
            block_labels=block_labels,
            base_block_counts=base_block_counts,
            base_block_indices=base_block_indices,
            real_total_len=real_total_len,
            padded_seqlen=padded_seqlen,
            is_real=is_real,
            pad_indices=pad_indices,
            sorted_real_indices=sort_idx,
            sorted_real_indices_i32=sort_idx.to(torch.int32),
            label_offsets_i32=torch.cat(
                (
                    torch.zeros(1, dtype=torch.int32, device=device),
                    counts_per_label.to(torch.int32).cumsum(dim=0, dtype=torch.int32),
                )
            ),
            expand_indices=expand_indices,
            gather_indices=gather_indices,
        )
    )


def _cube_topk_selection(q_real, k_real, precomputed, topk_ratio):
    layout = precomputed.layout
    dim = q_real.shape[-1]

    if q_real.device.type in {"cuda", "musa"} and q_real.dtype is torch.bfloat16 and k_real.dtype is torch.bfloat16 and q_real.is_contiguous() and k_real.is_contiguous():
        from .moorcat_h3_cube_pool import cube_pool_qk

        q_pool, k_pool = cube_pool_qk(
            q_real,
            k_real,
            layout.sorted_real_indices_i32,
            layout.label_offsets_i32,
        )
    else:
        q_sorted = q_real.index_select(0, layout.sorted_real_indices)
        k_sorted = k_real.index_select(0, layout.sorted_real_indices)
        label_lengths = layout.label_lengths
        token_labels = precomputed.runtime.token_labels
        if token_labels is None:
            token_labels = torch.repeat_interleave(
                torch.arange(label_lengths.numel(), device=q_real.device),
                label_lengths,
            )
            precomputed.runtime.token_labels = token_labels
        q_pool = torch.zeros(
            label_lengths.numel(),
            *q_sorted.shape[1:],
            dtype=q_sorted.dtype,
            device=q_sorted.device,
        )
        k_pool = torch.zeros_like(q_pool)
        q_pool.index_add_(0, token_labels, q_sorted)
        k_pool.index_add_(0, token_labels, k_sorted)
        normalizer = label_lengths.float().view(-1, 1, 1)
        q_pool /= normalizer
        k_pool /= normalizer

    scores = torch.einsum("lhd,mhd->hlm", q_pool, k_pool) * (dim**-0.5)

    candidate_mask = layout.topk_mask
    sparse_sizes = candidate_mask.sum(dim=-1)
    sparse_labels = sparse_sizes > 0
    scores.masked_fill_(~candidate_mask.unsqueeze(0), float("-inf"))

    selected_counts = (sparse_sizes.to(torch.float32) * topk_ratio).to(torch.long)
    selected_counts.clamp_(min=1)
    selected_counts = torch.minimum(selected_counts, sparse_sizes)
    selected_counts = torch.where(sparse_labels, selected_counts, torch.zeros_like(selected_counts))

    selected_order = torch.topk(
        scores,
        k=layout.topk_semantic_capacity,
        dim=-1,
        largest=True,
        sorted=True,
    ).indices
    return selected_order, selected_counts


def cube_topk_block_indices(q_real, k_real, precomputed, topk_ratio):
    """Expand semantic top-k choices into physical Moorcat block rows."""
    layout = precomputed.layout
    runtime = precomputed.runtime

    selected_order, selected_counts = _cube_topk_selection(q_real, k_real, precomputed, topk_ratio)
    semantic_capacity = selected_order.shape[-1]
    selected_semantic = selected_order
    semantic_rank = torch.arange(semantic_capacity, device=selected_order.device).view(1, 1, -1)
    selected_valid = semantic_rank < selected_counts.view(1, -1, 1)

    block_labels = layout.block_labels
    selected_semantic = selected_semantic.index_select(1, block_labels)
    selected_valid = selected_valid.index_select(1, block_labels).expand_as(selected_semantic)
    q_semantic = block_labels.view(1, -1, 1)
    # Base edges must not be repeated in the selected KV prefix.
    selected_valid = selected_valid & ~layout.base_block_mask[q_semantic, selected_semantic]

    label_block_indices = layout.label_block_indices
    label_block_counts = layout.label_block_counts
    max_label_blocks = layout.max_label_block_count
    selected_physical = label_block_indices[selected_semantic]
    physical_rank = torch.arange(max_label_blocks, device=selected_order.device).view(1, 1, 1, -1)
    selected_physical_valid = selected_valid.unsqueeze(-1) & (physical_rank < label_block_counts[selected_semantic].unsqueeze(-1))
    selected_physical = selected_physical.flatten(-2)
    selected_physical_valid = selected_physical_valid.flatten(-2)

    num_heads = q_real.shape[1]
    num_blocks = layout.num_blocks
    base_counts = layout.base_block_counts.expand(1, num_heads, -1)[0]
    base_indices = layout.base_block_indices.expand(1, num_heads, -1, -1)[0]
    sparse_q_blocks = layout.sparse_label_mask.index_select(0, layout.block_labels)
    base_counts = torch.where(sparse_q_blocks.view(1, -1), base_counts, torch.ones_like(base_counts))
    base_indices = base_indices[..., : runtime.sparse_base_capacity]
    base_rank = torch.arange(base_indices.shape[-1], device=q_real.device).view(1, 1, -1)
    base_valid = base_rank < base_counts.unsqueeze(-1)

    # Block ids fit in int32; using it halves the largest temporary allocation.
    candidate_indices = torch.cat((base_indices.to(torch.int32), selected_physical.to(torch.int32)), dim=-1)
    candidate_valid = torch.cat((base_valid, selected_physical_valid), dim=-1)
    # MUSA cannot stably sort this int32 shape, so compact with prefix positions.
    packed_width = candidate_indices.shape[-1]
    compact_position = candidate_valid.to(torch.long).cumsum(dim=-1).sub_(1)
    compact_position.clamp_(min=0, max=packed_width - 1)
    compact_source = torch.where(candidate_valid, candidate_indices, torch.zeros_like(candidate_indices))
    packed_kv = torch.zeros(
        *candidate_indices.shape[:-1],
        packed_width,
        dtype=torch.int32,
        device=q_real.device,
    )
    packed_kv.scatter_add_(-1, compact_position, compact_source)

    buffer = runtime.kv_indices_buffer
    output_q_blocks = num_blocks
    expected_prefix = (1, num_heads, output_q_blocks)
    if buffer is None or tuple(buffer.shape[:-1]) != expected_prefix or buffer.shape[-1] < packed_width:
        runtime.kv_num_blocks_buffer = torch.empty(1, num_heads, output_q_blocks, dtype=torch.int32, device=q_real.device)
        runtime.kv_indices_buffer = torch.empty(
            1,
            num_heads,
            output_q_blocks,
            packed_width,
            dtype=torch.int32,
            device=q_real.device,
        )

    kv_counts = runtime.kv_num_blocks_buffer
    kv_indices = runtime.kv_indices_buffer[..., :packed_width]
    selected_physical_counts = selected_physical_valid.sum(dim=-1, dtype=torch.int32)
    kv_counts[0, :, :num_blocks].copy_(base_counts + selected_physical_counts)
    kv_indices[0, :, :num_blocks].copy_(packed_kv.to(torch.int32))
    return kv_indices, kv_counts
