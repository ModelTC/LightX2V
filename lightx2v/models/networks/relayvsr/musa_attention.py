"""Exact RelayVSR windows through MUSA's mask-capable native SDPA.

Like SwiftVR, gather spatial neighborhoods into batched dense attention inputs.
Unlike SwiftVR's shared windows, every RelayVSR query keeps its own rectangle.
All condition/video planes participate in ONE softmax; GQA is not expanded.
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F

from .attention import SpatialAttentionSpec

_TILE = 16
_CACHE_BYTES = 3 * 1024**3
_GATHER_BYTES = 2 * 1024**3
_layouts = OrderedDict()
_layout_bytes = 0


def clear_layout_cache():
    global _layout_bytes
    _layouts.clear()
    _layout_bytes = 0


def _layout(height, width, windows, shifted, chunk, device):
    global _layout_bytes
    cache_key = (height, width, windows, shifted, chunk, device.type, device.index)
    if cache_key in _layouts:
        _layouts.move_to_end(cache_key)
        return _layouts[cache_key][0]

    tile_y, tile_x = torch.meshgrid(torch.arange(0, height, _TILE, device=device), torch.arange(0, width, _TILE, device=device), indexing="ij")
    tile_y, tile_x = tile_y.flatten(), tile_x.flatten()
    qr = torch.arange(_TILE * _TILE, device=device)
    parts, storage_bytes = [], 0
    for start in range(0, tile_y.numel(), chunk):
        y, x = tile_y[start : start + chunk, None], tile_x[start : start + chunk, None]
        qy, qx = y + qr[None] // _TILE, x + qr[None] % _TILE
        qi = (qy * width + qx).clamp(0, height * width - 1).flatten()
        keys, masks = [], []
        for frame, (wh, ww) in enumerate(windows):
            rows, cols = wh + _TILE - 1, ww + _TILE - 1
            if shifted:
                rows, cols = min(height, rows), min(width, cols)
            kr = torch.arange(rows * cols, device=device)
            top, left = y - wh // 2, x - ww // 2
            qtop, qleft = qy - wh // 2, qx - ww // 2
            if shifted:
                top, left = top.clamp(0, height - wh), left.clamp(0, width - ww)
                qtop, qleft = qtop.clamp(0, height - wh), qleft.clamp(0, width - ww)
            ky, kx = top + kr[None] // cols, left + kr[None] % cols
            valid = (ky >= 0) & (ky < height) & (kx >= 0) & (kx < width)
            keys.append(frame * height * width + (ky * width + kx).clamp(0, height * width - 1))
            masks.append(valid[:, None] & (ky[:, None] >= qtop[:, :, None]) & (ky[:, None] < qtop[:, :, None] + wh) & (kx[:, None] >= qleft[:, :, None]) & (kx[:, None] < qleft[:, :, None] + ww))
        ki = torch.cat(keys, -1)
        mask = torch.cat(masks, -1)[:, None].contiguous()
        owner = destination = None
        if height % _TILE or width % _TILE:
            owner = ((qy < height) & (qx < width)).flatten().nonzero().flatten()
            destination = qi.index_select(0, owner)
        part = (qi, ki.flatten(), mask, destination, owner, qy.shape[0], ki.shape[1])
        parts.append(part)
        storage_bytes += sum(t.numel() * t.element_size() for t in part if isinstance(t, torch.Tensor))

    # Bound persistent device storage across warmup resolutions and requests.
    while _layouts and _layout_bytes + storage_bytes > _CACHE_BYTES:
        _, (_, old_bytes) = _layouts.popitem(last=False)
        _layout_bytes -= old_bytes
    if storage_bytes <= _CACHE_BYTES:
        _layouts[cache_key] = parts, storage_bytes
        _layout_bytes += storage_bytes
    return parts


def _local_attention(query, key, value, height, width, windows, shifted):
    b, heads, tokens, dim = query.shape
    kv_heads = key.shape[1]
    if tokens != height * width or key.shape != value.shape or key.shape[0] != b or key.shape[2] != len(windows) * tokens or key.shape[3] != dim or heads % kv_heads:
        raise ValueError("Invalid RelayVSR local attention Q/K/V shape")
    if query.dtype != key.dtype or key.dtype != value.dtype or query.device != key.device or key.device != value.device:
        raise ValueError("RelayVSR local attention Q/K/V must share dtype and device")
    lengths = [(min(height, wh + _TILE - 1) * min(width, ww + _TILE - 1) if shifted else (wh + _TILE - 1) * (ww + _TILE - 1)) for wh, ww in windows]
    bytes_per_tile = sum(lengths) * kv_heads * dim * key.element_size() * 2
    limit = max(1, min(128, _GATHER_BYTES // bytes_per_tile))
    chunk = 1 << (limit.bit_length() - 1)
    parts = _layout(height, width, windows, shifted, chunk, query.device)
    output = query.new_empty((b, tokens, heads, dim))
    for batch in range(b):
        # Sequence-major storage makes gathers and the native BSHD output cheap.
        qs, ks, vs = (t[batch].transpose(0, 1).contiguous() for t in (query, key, value))
        for qi, ki, mask, destination, owner, count, key_length in parts:
            q = qs.index_select(0, qi).view(count, _TILE * _TILE, heads, dim).transpose(1, 2)
            k = ks.index_select(0, ki).view(count, key_length, kv_heads, dim).transpose(1, 2)
            v = vs.index_select(0, ki).view(count, key_length, kv_heads, dim).transpose(1, 2)
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=heads != kv_heads)
            attended = attended.transpose(1, 2).reshape(-1, heads, dim)
            if owner is not None:
                attended = attended.index_select(0, owner)
            output[batch].index_copy_(0, qi if destination is None else destination, attended)
    return output.transpose(1, 2)


@torch.library.custom_op("lightx2v::relayvsr_musa_flash_attention", mutates_args=(), device_types="cuda")
def _compiled_flash(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, height: int, width: int, windows: list[int]) -> torch.Tensor:
    windows = tuple(zip(windows[::2], windows[1::2]))
    if len(windows) > 1:
        frames = key.shape[2] // (height * width)
        windows = windows[:2] + (windows[2],) * (frames - 2)
    return musa_flash_attention(query, key, value, height, width, windows)


@_compiled_flash.register_fake
def _compiled_flash_fake(query, key, value, height, width, windows):
    b, heads, tokens, dim = query.shape
    return query.new_empty((b, tokens, heads, dim)).transpose(1, 2)


def musa_flash_attention(query, key, value, height, width, windows):
    if torch.compiler.is_compiling():
        return _compiled_flash(query, key, value, height, width, [size for window in windows for size in window])
    windows = tuple(tuple(window) for window in windows)
    if any(size <= 0 or size % 2 for window in windows for size in window):
        raise ValueError("RelayVSR Flash requires positive even window dimensions")
    return _local_attention(query, key, value, height, width, windows, shifted=False)


@torch.library.custom_op("lightx2v::relayvsr_musa_wan_attention", mutates_args=(), device_types="cuda")
def _compiled_wan(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, geometry: list[int]) -> torch.Tensor:
    return musa_spatial_attention(query, key, value, SpatialAttentionSpec(*geometry))


@_compiled_wan.register_fake
def _compiled_wan_fake(query, key, value, geometry):
    return torch.empty_like(query, memory_format=torch.contiguous_format)


def musa_spatial_attention(query, key, value, spec):
    if torch.compiler.is_compiling():
        geometry = [spec.frames, spec.height, spec.width, spec.window_height, spec.window_width, spec.temporal_window, spec.key_frames, spec.query_frame_offset, spec.key_frame_offset]
        return _compiled_wan(query, key, value, geometry)
    n = spec.height * spec.width
    first = spec.query_frame_offset - spec.key_frame_offset
    wh, ww = min(spec.height, spec.window_height), min(spec.width, spec.window_width)
    if min(wh, ww, spec.temporal_window) <= 0 or first < 0 or first + spec.frames > spec.key_frames or query.shape[1] != spec.frames * n or key.shape[1] != spec.key_frames * n:
        raise ValueError("Invalid RelayVSR Wan spatial/temporal geometry")
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    for frame in range(spec.frames):
        last = first + frame + 1
        start = max(0, last - spec.temporal_window)
        q = query[:, frame * n : (frame + 1) * n].transpose(1, 2)
        k, v = (t[:, start * n : last * n].transpose(1, 2) for t in (key, value))
        attended = _local_attention(q, k, v, spec.height, spec.width, ((wh, ww),) * (last - start), shifted=True)
        output[:, frame * n : (frame + 1) * n].copy_(attended.transpose(1, 2))
    return output
