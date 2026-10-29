"""RelayVSR attention geometry, shared SDPA windows and backend selection."""

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn.functional as F

LOCAL_ATTENTION_BACKENDS = ("triton", "torch_sdpa")


@dataclass(frozen=True)
class SpatialAttentionSpec:
    frames: int
    height: int
    width: int
    window_height: int
    window_width: int
    temporal_window: int
    key_frames: int
    query_frame_offset: int = 0
    key_frame_offset: int = 0


def get_local_attention_ops(backend="triton"):
    if backend == "torch_sdpa":
        from .flash_model_attention import flash_model_sdpa_attention
        from .wan_attention import wan_sdpa_attention

        return flash_model_sdpa_attention, wan_sdpa_attention

    if backend != "triton":
        raise ValueError(f"RelayVSR attention_backend must be one of {LOCAL_ATTENTION_BACKENDS}, got {backend!r}")

    from .flash_model_attention import flash_model_triton_attention
    from .wan_attention import wan_triton_attention

    return flash_model_triton_attention, wan_triton_attention


_TILE = 16
_CACHE_BYTES = 3 * 1024**3
_GATHER_BYTES = 2 * 1024**3
_layouts = OrderedDict()
_layout_bytes = 0


def clear_layout_cache():
    global _layout_bytes
    _layouts.clear()
    _layout_bytes = 0


def _layout(height, width, windows, shifted, chunk, device, query_start=0, query_tokens=None):
    global _layout_bytes
    query_tokens = height * width if query_tokens is None else query_tokens
    cache_key = (height, width, windows, shifted, chunk, device.type, device.index, query_start, query_tokens)
    if cache_key in _layouts:
        _layouts.move_to_end(cache_key)
        return _layouts[cache_key][0]

    if query_start == 0 and query_tokens == height * width:
        tile_y, tile_x = torch.meshgrid(torch.arange(0, height, _TILE, device=device), torch.arange(0, width, _TILE, device=device), indexing="ij")
        tile_y, tile_x = tile_y.flatten(), tile_x.flatten()
    else:
        tiles = [
            (y, x)
            for y in range(0, height, _TILE)
            for x in range(0, width, _TILE)
            if any(max(row * width + x, query_start) < min(row * width + min(x + _TILE, width), query_start + query_tokens) for row in range(y, min(y + _TILE, height)))
        ]
        tile_y = torch.tensor([y for y, _ in tiles], device=device)
        tile_x = torch.tensor([x for _, x in tiles], device=device)
    qr = torch.arange(_TILE * _TILE, device=device)
    parts, storage_bytes = [], 0
    for start in range(0, tile_y.numel(), chunk):
        y, x = tile_y[start : start + chunk, None], tile_x[start : start + chunk, None]
        qy, qx = y + qr[None] // _TILE, x + qr[None] % _TILE
        local_query = qy * width + qx - query_start
        qi = local_query.clamp(0, query_tokens - 1).flatten()
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
        if height % _TILE or width % _TILE or query_start != 0 or query_tokens != height * width:
            owner = ((qy < height) & (qx < width) & (local_query >= 0) & (local_query < query_tokens)).flatten().nonzero().flatten()
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


def _local_attention(query, key, value, height, width, windows, shifted, query_start=0):
    b, heads, tokens, dim = query.shape
    kv_heads = key.shape[1]
    if (
        query_start < 0
        or tokens < 1
        or query_start + tokens > height * width
        or key.shape != value.shape
        or key.shape[0] != b
        or key.shape[2] != len(windows) * height * width
        or key.shape[3] != dim
        or heads % kv_heads
    ):
        raise ValueError("Invalid RelayVSR local attention Q/K/V shape")
    if query.dtype != key.dtype or key.dtype != value.dtype or query.device != key.device or key.device != value.device:
        raise ValueError("RelayVSR local attention Q/K/V must share dtype and device")
    lengths = [(min(height, wh + _TILE - 1) * min(width, ww + _TILE - 1) if shifted else (wh + _TILE - 1) * (ww + _TILE - 1)) for wh, ww in windows]
    bytes_per_tile = sum(lengths) * kv_heads * dim * key.element_size() * 2
    limit = max(1, min(128, _GATHER_BYTES // bytes_per_tile))
    chunk = 1 << (limit.bit_length() - 1)
    parts = _layout(height, width, windows, shifted, chunk, query.device, query_start, tokens)
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
