"""VDN window attention for a complete sequence and a local head shard.

Adapted from OpenVDN/vdn-minimax-h3 e02ff077 (Apache-2.0).
"""

from dataclasses import dataclass
from functools import cache, lru_cache

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate


@dataclass(frozen=True)
class VDNLayout:
    sequence_length: int
    video_start: int
    num_frames: int
    frame_size: tuple[int, int]
    text_length: int
    chunk: int = 5
    radius: int = 1

    @property
    def tokens_per_frame(self):
        return self.frame_size[0] * self.frame_size[1]

    @property
    def video_end(self):
        return self.video_start + self.num_frames * self.tokens_per_frame

    @property
    def bounds(self):
        return tuple(((frame // self.chunk - self.radius) * self.chunk, (frame // self.chunk + self.radius + 1) * self.chunk - 1) for frame in range(self.num_frames))

    @property
    def full_cover(self):
        return all(lo <= 0 and hi >= self.num_frames - 1 for lo, hi in self.bounds)


def build_window_block_mask(layout, device, block_size=128):
    """Build block sparsity directly from frame windows."""
    start, end, spatial, frames = layout.video_start, layout.video_end, layout.tokens_per_frame, layout.num_frames
    chunk, radius = layout.chunk, layout.radius

    def mask_mod(batch, head, query, key):
        query_video = (query >= start) & (query < end)
        key_video = (key >= start) & (key < end)
        q_frame = ((query - start) // spatial).clamp(0, frames - 1)
        k_frame = (key - start) // spatial
        lo, hi = (q_frame // chunk - radius) * chunk, (q_frame // chunk + radius + 1) * chunk - 1
        window = (k_frame >= lo) & (k_frame <= hi)
        anchors = (q_frame == 0) | (q_frame == frames - 1) | (k_frame == 0) | (k_frame == frames - 1)
        return (~(query_video & key_video)) | window | anchors

    count = (layout.sequence_length + block_size - 1) // block_size
    columns = torch.arange(count)
    k_start = columns * block_size
    k_end = (k_start + block_size).clamp(max=layout.sequence_length)
    partial_indices = torch.zeros((1, 1, count, count), dtype=torch.int32)
    full_indices = torch.zeros_like(partial_indices)
    partial_count = torch.zeros((1, 1, count), dtype=torch.int32)
    full_count = torch.zeros_like(partial_count)
    # Global tokens/anchors are dense. Other rows use the union/intersection
    # of their frame windows for partial/full blocks.
    normal_start, normal_end = start + spatial, end - spatial
    for row in range(count):
        q_start, q_end = row * block_size, min((row + 1) * block_size, layout.sequence_length)
        first, last = max(q_start, normal_start), min(q_end, normal_end)
        if first >= last:
            any_allowed = torch.ones(count, dtype=torch.bool)
            all_allowed = any_allowed.clone()
        else:
            first_frame, last_frame = (first - start) // spatial, (last - 1 - start) // spatial
            union_lo = start + (first_frame // chunk - radius) * chunk * spatial
            union_hi = start + (last_frame // chunk + radius + 1) * chunk * spatial
            intersect_lo = max(normal_start, min(normal_end, start + (last_frame // chunk - radius) * chunk * spatial))
            intersect_hi = max(normal_start, min(normal_end, start + (first_frame // chunk + radius + 1) * chunk * spatial))
            any_allowed = (k_start < normal_start) | (k_end > normal_end) | ((k_start < union_hi) & (k_end > union_lo))
            if q_start < first or q_end > last:
                any_allowed.fill_(True)
            left_gap = (intersect_lo > normal_start) & (k_start < intersect_lo) & (k_end > normal_start)
            right_gap = (intersect_hi < normal_end) & (k_start < normal_end) & (k_end > intersect_hi)
            all_allowed = ~(left_gap | right_gap)
        # Match the token mask's zero padding in the last block.
        all_allowed &= (k_end - k_start == block_size) & (q_end - q_start == block_size)
        partial = columns[any_allowed & ~all_allowed]
        full = columns[all_allowed]
        partial_count[0, 0, row] = partial.numel()
        full_count[0, 0, row] = full.numel()
        partial_indices[0, 0, row, : partial.numel()] = partial
        full_indices[0, 0, row, : full.numel()] = full
    stats = {
        "block_size": block_size,
        "grid_blocks": count * count,
        "partial_blocks": int(partial_count.sum()),
        "full_blocks": int(full_count.sum()),
        "index_bytes": sum(t.numel() * t.element_size() for t in (partial_count, partial_indices, full_count, full_indices)),
    }
    stats["active_block_fraction"] = (stats["partial_blocks"] + stats["full_blocks"]) / stats["grid_blocks"]
    mask = BlockMask.from_kv_blocks(
        partial_count,
        partial_indices,
        full_count,
        full_indices,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
        seq_lengths=(layout.sequence_length, layout.sequence_length),
        compute_q_blocks=False,
    )
    return mask.to(device), stats


@lru_cache(maxsize=2)
def _shared_window_mask(layout, device):
    return build_window_block_mask(layout, torch.device(device))


@cache
def _compiled_window():
    return torch.compile(flex_attention, dynamic=False, fullgraph=True)


@ATTN_WEIGHT_REGISTER("vdn_window")
class VDNWindowAttention(AttnWeightTemplate):
    def __init__(self):
        self.config = {}
        self.backend = "flex_triton"
        self.layout = None
        self.mask = None
        self._mask_key = None
        self.block_stats = None

    def prepare(self, layout, device):
        self.layout = layout
        device = torch.device(device)
        if device.type != "cuda":
            self.backend = "sdpa_reference"
            return
        self.backend = "flex_triton"
        key = (layout, str(device))
        if key == self._mask_key:
            return
        self.mask, self.block_stats = _shared_window_mask(*key)
        self._mask_key = key

    def apply(self, q, k, v, **kwargs):
        if q.shape[0] != self.layout.sequence_length or k.shape[0] != self.layout.sequence_length:
            raise ValueError("VDN window attention needs the complete packed sequence on each head rank")
        if q.is_cuda:
            out = _compiled_window()(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], block_mask=self.mask, scale=q.shape[-1] ** -0.5, kernel_options={"BACKEND": "TRITON"})
            return out[0].transpose(0, 1)
        return self._reference(q, k, v)

    def _reference(self, q, k, v):
        layout = self.layout
        output = torch.empty_like(q)
        global_rows = torch.cat((torch.arange(layout.video_start, device=q.device), torch.arange(layout.video_end, layout.sequence_length, device=q.device)))

        def attend(query, rows):
            return F.scaled_dot_product_attention(query.transpose(0, 1)[None], k[rows].transpose(0, 1)[None], v[rows].transpose(0, 1)[None])[0].transpose(0, 1)

        if global_rows.numel():
            output[global_rows] = attend(q[global_rows], torch.arange(layout.sequence_length, device=q.device))
        for frame, (lo, hi) in enumerate(layout.bounds):
            if frame in (0, layout.num_frames - 1):
                lo, hi = 0, layout.num_frames - 1
            lo, hi = max(0, lo), min(layout.num_frames - 1, hi)
            extras = [f for f in (0, layout.num_frames - 1) if not lo <= f <= hi]
            rows = [global_rows, torch.arange(layout.video_start + lo * layout.tokens_per_frame, layout.video_start + (hi + 1) * layout.tokens_per_frame, device=q.device)]
            rows.extend(torch.arange(layout.video_start + f * layout.tokens_per_frame, layout.video_start + (f + 1) * layout.tokens_per_frame, device=q.device) for f in extras)
            start = layout.video_start + frame * layout.tokens_per_frame
            output[start : start + layout.tokens_per_frame] = attend(q[start : start + layout.tokens_per_frame], torch.cat(rows))
        return output
