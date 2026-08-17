# Copyright 2026 Lightricks Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tiling geometry used only by the LTX-2.5 diffusion VAE decoder.

The older LTX-2.x ConvVAE tiles its latent input before the whole decoder.
DiffVAE instead runs stages 1--3 once, tiles the stage-4 feature volume, then
blends pixels produced by stages 4--5.  Keeping this module separate prevents
the different temporal/drop-leading semantics from changing LTX-2.3.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import torch

from lightx2v.models.video_encoders.hf.ltx2.video_vae.tiling import (
    Tile,
    TilingConfig,
    compute_trapezoidal_mask_1d,
)


@dataclass(frozen=True)
class AxisInterval:
    start: int
    end: int
    left_ramp: int = 0
    right_ramp: int = 0


def _split_axis(length: int, size: int, overlap: int, min_size: int) -> list[AxisInterval]:
    if length <= size:
        return [AxisInterval(0, length)]
    if size <= overlap:
        raise ValueError(f"DiffVAE tile size must exceed overlap, got size={size}, overlap={overlap}")

    step = size - overlap
    starts = list(range(0, max(1, length - overlap), step))
    intervals = [
        AxisInterval(
            start=start,
            end=min(start + size, length),
            left_ramp=overlap if index else 0,
            right_ramp=overlap if start + size < length else 0,
        )
        for index, start in enumerate(starts)
    ]
    if len(intervals) > 1 and intervals[-1].end - intervals[-1].start < min_size:
        last = intervals[-1]
        new_start = max(0, last.end - min_size)
        previous = intervals[-2]
        new_overlap = previous.end - new_start
        intervals[-2] = replace(previous, right_ramp=new_overlap)
        intervals[-1] = replace(last, start=new_start, left_ramp=new_overlap)
    return intervals


def _axis_specs(
    length: int,
    tile_size: int,
    overlap: int,
    min_size: int,
    scale: int,
    *,
    temporal: bool,
) -> list[tuple[slice, slice, torch.Tensor]]:
    intervals = _split_axis(length, tile_size, overlap, min_size)
    specs = []
    for interval in intervals:
        out_start = interval.start * scale
        out_end = interval.end * scale
        left_ramp = interval.left_ramp * scale
        right_ramp = interval.right_ramp * scale
        if temporal and scale == 2:
            # PixelShuffle drops the duplicated leading frame only for the
            # origin tile.  Non-origin tiles retain it for overlap blending.
            out_end -= 1
            if interval.start:
                out_start -= 1
        mask = compute_trapezoidal_mask_1d(
            out_end - out_start,
            left_ramp,
            right_ramp,
            left_starts_from_0=False,
        )
        specs.append((slice(interval.start, interval.end), slice(out_start, out_end), mask))
    return specs


def prepare_tile_schedule(
    stage4_shape: torch.Size,
    tiling_config: TilingConfig,
    *,
    pixel_height: int,
    pixel_width: int,
    upsample_stride: tuple[int, int, int],
    patch_size: int,
    min_tile_size: tuple[int, int, int],
) -> list[Tile]:
    """Return source-compatible stage-4 input / pixel-output tile pairs."""
    _, stage4_t, stage4_h, stage4_w, _ = stage4_shape
    scale_t = upsample_stride[0]
    scale_h = upsample_stride[1] * patch_size
    scale_w = upsample_stride[2] * patch_size

    temporal = tiling_config.temporal_config
    if temporal is None:
        t_size, t_overlap = stage4_t, 0
    else:
        t_size = max(min_tile_size[0], temporal.tile_size_in_frames // scale_t)
        t_overlap = temporal.tile_overlap_in_frames // scale_t
        t_size = max(t_size, 2 * t_overlap)

    spatial = tiling_config.spatial_config
    if spatial is None:
        h_size, w_size, spatial_overlap = stage4_h, stage4_w, 0
    else:
        long_pixels = max(pixel_height, pixel_width)
        long_stage4 = max(stage4_h, stage4_w)
        requested = spatial.tile_size_in_pixels // max(scale_h, scale_w)
        h_size = max(min_tile_size[1], round(requested * stage4_h / long_stage4))
        w_size = max(min_tile_size[2], round(requested * stage4_w / long_stage4))
        spatial_overlap = spatial.tile_overlap_in_pixels // max(scale_h, scale_w)
        h_size = max(h_size, 2 * spatial_overlap)
        w_size = max(w_size, 2 * spatial_overlap)
        # A full-size request should remain exactly untiled on that axis.
        if spatial.tile_size_in_pixels >= long_pixels:
            h_size, w_size = stage4_h, stage4_w

    t_specs = _axis_specs(stage4_t, t_size, t_overlap, min_tile_size[0], scale_t, temporal=True)
    h_specs = _axis_specs(stage4_h, h_size, spatial_overlap, min_tile_size[1], scale_h, temporal=False)
    w_specs = _axis_specs(stage4_w, w_size, spatial_overlap, min_tile_size[2], scale_w, temporal=False)

    one = torch.ones(1)
    tiles: list[Tile] = []
    for t_spec, h_spec, w_spec in itertools.product(t_specs, h_specs, w_specs):
        t_in, t_out, t_mask = t_spec
        h_in, h_out, h_mask = h_spec
        w_in, w_out, w_mask = w_spec
        tiles.append(
            Tile(
                in_coords=(slice(None), t_in, h_in, w_in, slice(None)),
                out_coords=(slice(None), slice(None), t_out, h_out, w_out),
                masks_1d=(one, one, t_mask, h_mask, w_mask),
            )
        )
    return tiles


def group_tiles_by_temporal_slice(tiles: list[Tile]) -> list[list[Tile]]:
    groups: list[list[Tile]] = []
    for tile in tiles:
        if not groups or groups[-1][0].out_coords[2] != tile.out_coords[2]:
            groups.append([tile])
        else:
            groups[-1].append(tile)
    return groups


def stage4_shape_from_latent(
    latent_t: int,
    latent_h: int,
    latent_w: int,
    strides: tuple[tuple[int, int, int], ...],
) -> tuple[int, int, int]:
    t, h, w = latent_t, latent_h, latent_w
    for stride_t, stride_h, stride_w in strides[:3]:
        t, h, w = t * stride_t, h * stride_h, w * stride_w
        if stride_t == 2:
            t -= 1
    return t, h, w


def tile_shape(full_shape: tuple[int, ...], coords: tuple[slice, ...]) -> tuple[int, ...]:
    return tuple(len(range(*coord.indices(size))) for size, coord in zip(full_shape, coords, strict=True))


def separable_mask(tile: Tile, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.ones(1, device=device, dtype=dtype)
    for dim, mask_1d in enumerate(tile.masks_1d):
        shape = [1] * len(tile.out_coords)
        shape[dim] = mask_1d.numel()
        mask = mask * mask_1d.to(device=device, dtype=dtype).view(shape)
    return mask


__all__ = [
    "group_tiles_by_temporal_slice",
    "prepare_tile_schedule",
    "separable_mask",
    "stage4_shape_from_latent",
    "tile_shape",
]
