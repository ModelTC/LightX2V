# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 semantic cube attention backed by Moorcat block-sparse kernels."""

from dataclasses import dataclass
from functools import lru_cache

import torch

from lightx2v_platform.registry_factory import PLATFORM_SPARSE_OPERATOR_REGISTER

from .moorcat_h3_mask import PackedStreams, cube_topk_block_indices, precompute_cube_attention

_SUPPORTED_CUBES = {(4, 4, 4): 64, (8, 4, 4): 128}


@dataclass
class _RequestState:
    key: tuple
    source_layout: object
    precomputed: object
    topk_ratio: float
    dense_original_indices: torch.Tensor
    dense_cube_indices: torch.Tensor
    kernel_cu: torch.Tensor
    kernel_len: int
    tile: int
    dense_cu_q: torch.Tensor
    dense_cu_k: torch.Tensor
    dense_q_len: int
    head_mask_type: torch.Tensor | None = None
    logged: bool = False


_STATE = None


@lru_cache(maxsize=1)
def _moorcat_kernel():
    from moorcat.blocksparse import block_sparse_attn_func_indexed_fast

    return block_sparse_attn_func_indexed_fast


@lru_cache(maxsize=1)
def _dense_kernel():
    from flash_attn_interface import flash_attn_varlen_func

    return flash_attn_varlen_func


def _topk_ratio(setting):
    ratio = float(setting["topk_ratio"])
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"topk_ratio must be in (0, 1], got {ratio}")
    return ratio


def _request_key(cube_size, topk_ratio):
    return tuple(cube_size), topk_ratio


def _packed_streams(layout):
    """Split packed modality indices into condition and target streams."""

    num_condition_video_rows = int(layout.num_condition_video_rows)
    num_condition_audio_rows = int(layout.num_condition_audio_rows)
    condition_video = layout.video_indices[:num_condition_video_rows]
    target_video = layout.video_indices[num_condition_video_rows:]
    condition_audio = layout.audio_indices[:num_condition_audio_rows]
    target_audio = layout.audio_indices[num_condition_audio_rows:]

    return PackedStreams(
        text=layout.text_indices,
        cond_image=condition_video,
        latent=target_video,
        cond_audio=condition_audio,
        audio=target_audio,
        cond_image_shapes=[layout.cond_image_shapes],
        cond_image_roles=[layout.cond_image_roles],
        cond_event_orders=[layout.cond_event_order],
        cond_audio_stream_lens=[layout.cond_audio_stream_lens],
    )


def _configure(scheduler, setting):
    global _STATE

    layout = scheduler.layout

    cube_size = tuple(int(value) for value in setting.get("local_cube_size", (8, 4, 4)))
    try:
        tile = _SUPPORTED_CUBES[cube_size]
    except KeyError as exc:
        raise ValueError(f"unsupported Moorcat cube size: {cube_size}") from exc

    topk_ratio = _topk_ratio(setting)
    key = _request_key(cube_size, topk_ratio)
    if _STATE is not None and _STATE.source_layout is layout and _STATE.key == key:
        return _STATE

    device = layout.position_ids.device
    patch_t, patch_h, patch_w = tuple(scheduler.config.get("patch_size", (1, 2, 2)))
    target_shape = (
        scheduler.num_latent_frames // patch_t,
        scheduler.latent_height // patch_h,
        scheduler.latent_width // patch_w,
    )
    precomputed = precompute_cube_attention(
        [target_shape],
        torch.tensor([0, layout.sequence_length], dtype=torch.long, device=device),
        layout.sequence_length,
        cube_size,
        device,
        _packed_streams(layout),
        layout.position_ids,
        topk_ratio if topk_ratio < 1.0 else 0.0,
    )
    cube_layout = precomputed.layout
    if int(cube_layout.label_lengths.max().item()) > tile:
        raise ValueError("musa_moorcat_h3_sparse requires each semantic cube to fit in one physical block")

    sparse_blocks = cube_layout.sparse_label_mask.index_select(0, cube_layout.block_labels)
    sparse_base_counts = cube_layout.base_block_counts[0, 0, sparse_blocks]
    precomputed.runtime.sparse_base_capacity = max(int(sparse_base_counts.max().item()), 1)

    sparse_tokens = sparse_blocks.repeat_interleave(tile)
    dense_cube_indices = torch.nonzero((~sparse_tokens) & cube_layout.is_real.bool(), as_tuple=False).squeeze(1)
    dense_original_indices = cube_layout.gather_indices.index_select(0, dense_cube_indices)

    kernel_len = cube_layout.padded_seqlen
    _STATE = _RequestState(
        key=key,
        source_layout=layout,
        precomputed=precomputed,
        topk_ratio=topk_ratio,
        dense_original_indices=dense_original_indices,
        dense_cube_indices=dense_cube_indices,
        kernel_cu=torch.tensor([0, kernel_len], dtype=torch.int32, device=device),
        kernel_len=kernel_len,
        tile=tile,
        dense_cu_q=torch.tensor([0, dense_original_indices.numel()], dtype=torch.int32, device=device),
        dense_cu_k=torch.tensor([0, cube_layout.real_total_len], dtype=torch.int32, device=device),
        dense_q_len=int(dense_original_indices.numel()),
    )
    print(
        "MINIMAX_H3_MOORCAT_CONFIGURED "
        f"path=native{tile} cube_size={cube_size} blocks={cube_layout.num_blocks} "
        f"real_len={cube_layout.real_total_len} kernel_len={kernel_len} "
        f"max_topk={cube_layout.topk_semantic_capacity}",
        flush=True,
    )
    return _STATE


def _dense_prefix(query, key, value, state, softmax_scale):
    dense_q = query.index_select(0, state.dense_original_indices)
    return _dense_kernel()(
        dense_q,
        key,
        value,
        cu_seqlens_q=state.dense_cu_q,
        cu_seqlens_k=state.dense_cu_k,
        max_seqlen_q=state.dense_q_len,
        max_seqlen_k=state.precomputed.layout.real_total_len,
        softmax_scale=softmax_scale,
        causal=False,
    )


def _dense_full(query, key, value, state, softmax_scale):
    cu = state.dense_cu_k
    return _dense_kernel()(
        query,
        key,
        value,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=state.precomputed.layout.real_total_len,
        max_seqlen_k=state.precomputed.layout.real_total_len,
        softmax_scale=softmax_scale,
        causal=False,
    )


@PLATFORM_SPARSE_OPERATOR_REGISTER("musa_moorcat_h3_sparse")
class MusaMoorcatH3SparseOperator:
    q_block_size = 128
    k_block_size = 128
    builds_block_indices = True

    def __init__(self, operator_setting):
        self.setting = operator_setting

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
        scheduler = kwargs["scheduler"]
        state = _configure(scheduler, self.setting)
        layout = state.precomputed.layout
        if q.shape[0] != layout.real_total_len or k.shape[0] != layout.real_total_len:
            raise ValueError(f"packed sequence changed after Moorcat setup: q={q.shape[0]} k={k.shape[0]} expected={layout.real_total_len}")

        ratio = state.topk_ratio
        softmax_scale = float(kwargs.get("softmax_scale", q.shape[-1] ** -0.5))
        if ratio == 1.0:
            return _dense_full(q, k, v, state, softmax_scale).reshape(q.shape[0], -1)

        indices, counts = cube_topk_block_indices(q, k, state.precomputed, ratio)
        cube_q = q.index_select(0, layout.gather_indices)
        cube_k = k.index_select(0, layout.gather_indices)
        cube_v = v.index_select(0, layout.gather_indices)
        if layout.pad_indices.numel():
            cube_q.index_fill_(0, layout.pad_indices, 0)
            cube_k.index_fill_(0, layout.pad_indices, 0)
            cube_v.index_fill_(0, layout.pad_indices, 0)

        dense_out = _dense_prefix(q, k, v, state, softmax_scale)
        indices = indices.contiguous()
        counts = counts.contiguous()
        num_heads = q.shape[1]
        if state.head_mask_type is None:
            state.head_mask_type = torch.ones(num_heads, dtype=torch.int32, device=q.device)

        out, _ = _moorcat_kernel()(
            cube_q,
            cube_k,
            cube_v,
            state.kernel_cu,
            state.head_mask_type,
            indices,
            counts,
            indices.shape[-1],
            state.kernel_len,
            softmax_scale,
            state.tile,
        )
        out.index_copy_(0, state.dense_cube_indices, dense_out)
        out = out.index_select(0, layout.expand_indices)
        if not state.logged:
            print(
                f"MINIMAX_H3_MOORCAT_ACTIVE path=native{state.tile} ratio={ratio} q={tuple(q.shape)} dense_q={state.dense_q_len} kernel_len={state.kernel_len} max_k={indices.shape[-1]}",
                flush=True,
            )
            state.logged = True
        return out.reshape(q.shape[0], -1)
