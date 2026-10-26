"""Released StreamRender-H3 window and packed-token geometry.

The online renderer schedules merged H3 latent units (2, 1, 1, 1), keeps a
three-latent sink, and packs each document as text, picture, reference, stereo
audio, then target video. All positions are computed on the official 40 Hz
clock before the transformer applies RoPE.
"""

from dataclasses import dataclass
from math import sqrt

import numpy as np
import torch

from lightx2v.models.networks.minimax_h3.packing import AUDIO_TAG, VIDEO_TAG

_UNIT_OFFSETS = (0, 2, 3, 4)
_FRAME_OFFSETS = (0, 1, 5, 9, 13)


def unit_boundary(index: int) -> int:
    """Map a merged scheduling-unit boundary to native H3 latents."""
    cycles, offset = divmod(index, 4)
    return 5 * cycles + _UNIT_OFFSETS[offset]


def frame_boundary(index: int) -> int:
    """Map a native H3 latent boundary to decoded 24 fps frames."""
    cycles, offset = divmod(index, 5)
    return 17 * cycles + _FRAME_OFFSETS[offset]


def audio_boundary(index: int) -> int:
    """First 40 Hz audio latent whose clock follows video latent ``index``."""
    return (5 * frame_boundary(index) + 2) // 3


def position_start(index: int) -> float:
    """Use the checkpoint's floating-point operation order for RoPE time."""
    cycles, offset = divmod(index, 5)
    return cycles * (17 * 5 / 3) + (5 / 3) * _FRAME_OFFSETS[offset]


def unit_index(boundary: int) -> int:
    cycles, offset = divmod(boundary, 5)
    if offset not in _UNIT_OFFSETS:
        raise ValueError(f"latent {boundary} is not a merged scheduling-unit boundary")
    return cycles * 4 + _UNIT_OFFSETS.index(offset)


def window_starts(num_frames: int) -> tuple[int, ...]:
    """Valid request boundaries for the release's 7-unit bootstrap and 2-unit chunks."""
    if num_frames <= 0:
        raise ValueError("StreamRender-H3 frame count must be positive")
    latent_count = next(index for index in range(1, num_frames + 1) if frame_boundary(index) >= num_frames)
    if frame_boundary(latent_count) != num_frames:
        raise ValueError("StreamRender-H3 frame count must end at a native H3 latent boundary")
    starts = [0]
    unit = 7
    while unit_boundary(unit) < latent_count:
        starts.append(unit_boundary(unit))
        unit += 2
    return tuple(starts)


@dataclass(frozen=True)
class StreamRenderH3WindowPlan:
    start: int
    stop: int
    sink_history_stop: int
    recent_start: int
    rope_window_stop: int
    video_indices: torch.Tensor
    reference_indices: torch.Tensor
    audio_indices: torch.Tensor
    video_noisy_mask: torch.Tensor
    audio_noisy_mask: torch.Tensor
    audio_commit_mask: torch.Tensor
    audio_commit_stop: int
    audio_prediction_stop: int
    target_latent_size: tuple[int, int]
    reference_latent_size: tuple[int, int]

    @property
    def is_bootstrap(self) -> bool:
        return self.start == 0

    @property
    def frame_start(self) -> int:
        return frame_boundary(self.start)

    @property
    def frame_stop(self) -> int:
        return frame_boundary(self.stop)

    @property
    def frame_count(self) -> int:
        return self.frame_stop - self.frame_start

    @property
    def target_frame_rows(self) -> int:
        height, width = self.target_latent_size
        return height * width // 4

    @property
    def reference_frame_rows(self) -> int:
        height, width = self.reference_latent_size
        return height * width // 4

    @property
    def noisy_video_indices(self) -> torch.Tensor:
        return self.video_indices[self.video_noisy_mask]

    @property
    def noisy_audio_indices(self) -> torch.Tensor:
        return self.audio_indices[self.audio_noisy_mask]


def build_window_plan(
    start: int,
    stop: int,
    *,
    target_latent_size: tuple[int, int],
    reference_latent_size: tuple[int, int],
    audio_length: int | None = None,
    previous_audio_stop: int | None = None,
) -> StreamRenderH3WindowPlan:
    """Select sink, rolling history and newly generated media for one window."""
    start_unit = unit_index(start)
    expected_stop = unit_boundary(7 if start == 0 else start_unit + 2)
    if start != 0 and (start_unit < 7 or (start_unit - 7) % 2):
        raise ValueError(f"latent {start} is not a StreamRender-H3 window start")
    if stop != expected_stop:
        raise ValueError(f"window from latent {start} must stop at latent {expected_stop}")
    sink_stop = unit_boundary(2)
    sink_history_stop = min(sink_stop, start)
    recent_start = unit_boundary(max(0, start_unit - 2))
    selected = list(range(stop)) if start == 0 else sorted(set(range(sink_history_stop)) | set(range(recent_start, stop)))
    video_indices = torch.tensor(selected, dtype=torch.long)
    video_noisy_mask = video_indices >= start
    audio_indices_list = []
    audio_commit_list = []
    for index in selected:
        lower, upper = audio_boundary(index), audio_boundary(index + 1)
        if audio_length is not None:
            lower, upper = min(lower, audio_length), min(upper, audio_length)
        audio_indices_list.extend(range(lower, upper))
        audio_commit_list.extend([index >= start] * (upper - lower))
    audio_commit_stop = audio_boundary(stop) if audio_length is None else min(audio_boundary(stop), audio_length)
    audio_prediction_stop = audio_commit_stop  # Released renderer has zero right lookahead.
    audio_indices = torch.tensor(audio_indices_list, dtype=torch.long)
    if previous_audio_stop is None:
        previous_audio_stop = 0 if start == 0 else audio_boundary(start)
    audio_noisy_mask = audio_indices >= previous_audio_stop
    return StreamRenderH3WindowPlan(
        start=start,
        stop=stop,
        sink_history_stop=sink_history_stop,
        recent_start=recent_start,
        rope_window_stop=unit_boundary(6),
        video_indices=video_indices,
        reference_indices=video_indices.clone(),  # keep_sink_reference=True
        audio_indices=audio_indices,
        video_noisy_mask=video_noisy_mask,
        audio_noisy_mask=audio_noisy_mask,
        audio_commit_mask=torch.tensor(audio_commit_list, dtype=torch.bool),
        audio_commit_stop=audio_commit_stop,
        audio_prediction_stop=audio_prediction_stop,
        target_latent_size=target_latent_size,
        reference_latent_size=reference_latent_size,
    )


def _spatial_grid(height: int, width: int) -> torch.Tensor:
    area = sqrt(height * width)
    axes = []
    for dim in (height, width):
        ratio = dim / area
        left = (1 - ratio) / 2
        axes.append(torch.from_numpy(np.linspace(left, left + ratio, dim // 2, endpoint=False) * 32).to(torch.float64))
    yy, xx = torch.meshgrid(*axes, indexing="ij")
    return torch.stack((yy.flatten(), xx.flatten()), dim=-1)


@dataclass(frozen=True)
class StreamRenderH3Layout:
    """One transformer's rows in native order or prefix-first cache order."""

    source_rows: torch.Tensor
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    text_indices: torch.Tensor
    picture_indices: torch.Tensor
    reference_indices: torch.Tensor
    audio_indices: torch.Tensor
    video_indices: torch.Tensor
    noisy_audio_indices: torch.Tensor
    noisy_video_indices: torch.Tensor
    prefix_length: int
    valid_sequence_length: int
    capture_prefix: bool
    q_ranges: torch.Tensor
    k_ranges: torch.Tensor


def build_window_layout(
    plan: StreamRenderH3WindowPlan,
    text_token_tags: torch.Tensor,
    *,
    picture_latent_size: tuple[int, int],
    prefix_cached: bool = False,
) -> StreamRenderH3Layout:
    """Build the fixed-window RoPE and prefix-attention layout for one NFE."""
    text_token_tags = text_token_tags.cpu().to(torch.long)
    text_count = text_token_tags.numel()
    picture_grid = _spatial_grid(*picture_latent_size)
    reference_grid = _spatial_grid(*plan.reference_latent_size)
    target_grid = _spatial_grid(*plan.target_latent_size)
    picture_count = picture_grid.shape[0]
    reference_count = plan.reference_indices.numel() * reference_grid.shape[0]
    audio_count = 2 * plan.audio_indices.numel()
    video_count = plan.video_indices.numel() * target_grid.shape[0]
    picture_start = text_count
    reference_start = picture_start + picture_count
    audio_start = reference_start + reference_count
    video_start = audio_start + audio_count
    total = video_start + video_count
    positions = torch.zeros((total, 3), dtype=torch.float64)
    positions[:text_count, 0] = torch.arange(text_count, dtype=torch.float64)
    positions[picture_start:reference_start, 0] = text_count
    positions[picture_start:reference_start, 1:] = picture_grid

    video_times = torch.tensor([position_start(int(index)) for index in plan.video_indices], dtype=torch.float64)
    reference_times = torch.tensor([position_start(int(index)) for index in plan.reference_indices], dtype=torch.float64)
    audio_times = plan.audio_indices.to(torch.float64)
    target_origin = float(text_count + 1)
    if plan.is_bootstrap:
        target_origin += position_start(plan.stop)
    else:
        sink_end = position_start(unit_boundary(2))
        recent_origin = position_start(max(unit_boundary(2), plan.recent_start))
        video_times = torch.where(plan.video_indices < plan.sink_history_stop, video_times, sink_end + video_times - recent_origin)
        reference_times = torch.where(plan.reference_indices < plan.sink_history_stop, reference_times, sink_end + reference_times - recent_origin)
        audio_times = torch.where(plan.audio_indices < audio_boundary(plan.sink_history_stop), audio_times, sink_end + audio_times - recent_origin)
        target_origin += position_start(plan.rope_window_stop)
    reference_positions = positions[reference_start:audio_start].view(plan.reference_indices.numel(), reference_grid.shape[0], 3)
    reference_positions[:, :, 0] = (text_count + 1 + reference_times)[:, None]
    reference_positions[:, :, 1:] = reference_grid[None]
    video_positions = positions[video_start:].view(plan.video_indices.numel(), target_grid.shape[0], 3)
    video_positions[:, :, 0] = (target_origin + video_times)[:, None]
    video_positions[:, :, 1:] = target_grid[None]
    audio_positions = positions[audio_start:video_start].view(2, -1, 3)
    audio_positions[:, :, 0] = target_origin + audio_times
    audio_positions[0, :, 2] = target_grid[:, 1].min()
    audio_positions[1, :, 2] = target_grid[:, 1].max()

    tags = torch.full((total,), VIDEO_TAG, dtype=torch.long)
    tags[:text_count] = text_token_tags
    tags[audio_start:video_start] = AUDIO_TAG
    prefix = torch.zeros(total, dtype=torch.bool)
    if not plan.is_bootstrap:
        prefix[:reference_start] = True
        reference_mask = plan.reference_indices < plan.sink_history_stop
        prefix[reference_start:audio_start] = reference_mask.repeat_interleave(reference_grid.shape[0])
        audio_mask = (plan.audio_indices < audio_boundary(plan.sink_history_stop)) & ~plan.audio_noisy_mask
        prefix[audio_start:video_start] = audio_mask.repeat(2)
        video_mask = (plan.video_indices < plan.sink_history_stop) & ~plan.video_noisy_mask
        prefix[video_start:] = video_mask.repeat_interleave(target_grid.shape[0])
    prefix_rows = prefix.nonzero().flatten()
    dynamic_rows = (~prefix).nonzero().flatten()
    source = dynamic_rows if prefix_cached else torch.cat((prefix_rows, dynamic_rows))
    destination = torch.full((total,), -1, dtype=torch.long)
    destination[source] = torch.arange(source.numel())

    def selected(rows: torch.Tensor) -> torch.Tensor:
        mapped = destination[rows]
        return mapped[mapped >= 0]

    text_rows = torch.arange(text_count)
    picture_rows = torch.arange(picture_start, reference_start)
    reference_rows = torch.arange(reference_start, audio_start)
    audio_rows = torch.arange(audio_start, video_start)
    video_rows = torch.arange(video_start, total)
    noisy_audio_rows = audio_rows[plan.audio_noisy_mask.repeat(2)]
    noisy_video_rows = video_rows[plan.video_noisy_mask.repeat_interleave(target_grid.shape[0])]
    prefix_length = prefix_rows.numel()
    valid_length = source.numel()
    if prefix_cached:
        q_ranges = torch.tensor([[0, valid_length]], dtype=torch.int32)
        k_ranges = torch.tensor([[0, prefix_length + valid_length]], dtype=torch.int32)
    elif prefix_length:
        q_ranges = torch.tensor([[0, prefix_length], [prefix_length, valid_length]], dtype=torch.int32)
        k_ranges = torch.tensor([[0, prefix_length], [0, valid_length]], dtype=torch.int32)
    else:
        q_ranges = torch.tensor([[0, valid_length]], dtype=torch.int32)
        k_ranges = q_ranges.clone()
    return StreamRenderH3Layout(
        source_rows=source,
        position_ids=positions[source],
        token_tags=tags[source],
        text_indices=selected(text_rows),
        picture_indices=selected(picture_rows),
        reference_indices=selected(reference_rows),
        audio_indices=selected(audio_rows),
        video_indices=selected(video_rows),
        noisy_audio_indices=selected(noisy_audio_rows),
        noisy_video_indices=selected(noisy_video_rows),
        prefix_length=prefix_length,
        valid_sequence_length=valid_length,
        capture_prefix=bool(prefix_length and not prefix_cached),
        q_ranges=q_ranges,
        k_ranges=k_ranges,
    )


__all__ = [
    "StreamRenderH3Layout",
    "StreamRenderH3WindowPlan",
    "audio_boundary",
    "build_window_layout",
    "build_window_plan",
    "frame_boundary",
    "position_start",
    "unit_boundary",
    "window_starts",
]
