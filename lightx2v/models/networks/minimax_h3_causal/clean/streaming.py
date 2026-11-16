"""CleanAR geometry ported from Zoe H3 v0.1.0, using native H3 packing."""

import math
from dataclasses import dataclass, replace

import torch

from lightx2v.models.networks.minimax_h3.packing import AUDIO_CHANNELS, AUDIO_TAG, VIDEO_TAG, MiniMaxH3PackedSequence, _temporal_position_grid
from lightx2v.models.networks.minimax_h3.packing_ref2av import MiniMaxH3PreparedReference, _frame_position_grid


@dataclass(frozen=True)
class CleanAttentionPolicy:
    num_frame_per_chunk: int = 5
    sink_chunk_num: int = 1
    latest_chunk_num: int = 2
    kv_cache_max_t_rope: int = 7
    sink_text_condition: bool = True
    sink_ref_condition: bool = True
    motion_prompt_origin: int = 20480
    media_origin: int = 30000

    def __post_init__(self):
        if self.num_frame_per_chunk != 5 or self.sink_chunk_num != 1 or self.latest_chunk_num != 2:
            raise ValueError("H3 v0.1.0 requires chunk=5, sink=1, latest=2")
        if self.motion_prompt_origin is None or self.media_origin is None:
            raise ValueError("On-demand motion encoding requires fixed motion_prompt_origin and media_origin")
        if self.kv_cache_max_t_rope < self.sink_chunk_num + self.latest_chunk_num - 1:
            raise ValueError("CleanAR rolling RoPE must keep sink and recent slots disjoint")


def build_chunk_specs(num_latent_frames, num_audio_latents, policy):
    times = _temporal_position_grid(num_latent_frames, 0.0)
    chunks = []
    audio_start = 0
    for start in range(0, num_latent_frames, policy.num_frame_per_chunk):
        end = min(start + policy.num_frame_per_chunk, num_latent_frames)
        if end == num_latent_frames:
            audio_end = num_audio_latents
        else:
            boundary = float(times[end])
            rounded = round(boundary)
            audio_end = min(rounded if math.isclose(boundary, rounded, abs_tol=1e-9, rel_tol=0) else math.ceil(boundary), num_audio_latents)
        chunks.append((start, end, audio_start, audio_end))
        audio_start = audio_end
    return chunks


CleanCacheKey = tuple[str, int]
GLOBAL_KEY: CleanCacheKey = ("global", -1)
REFERENCE_KEY: CleanCacheKey = ("reference", -1)


def prompt_key(index: int) -> CleanCacheKey:
    return "prompt", index


def chunk_key(index: int) -> CleanCacheKey:
    return "chunk", index


@dataclass(frozen=True)
class MiniMaxH3CleanStreamingPlan:
    """Condition blocks, prompt segments and media chunks for CleanAR."""

    global_position_ids: torch.Tensor
    global_token_tags: torch.Tensor
    reference_position_ids: torch.Tensor
    reference_token_tags: torch.Tensor
    num_reference_video_rows: int
    num_reference_audio_rows: int
    prompt_position_ids: tuple[torch.Tensor, ...]
    prompt_token_tags: tuple[torch.Tensor, ...]
    target_position_ids: torch.Tensor
    target_token_tags: torch.Tensor
    chunk_specs: torch.Tensor
    segment_starts: torch.Tensor
    num_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int
    policy: CleanAttentionPolicy
    patch_size: tuple[int, int, int] = (1, 2, 2)

    @property
    def rows_per_frame(self) -> int:
        return (self.latent_height // self.patch_size[1]) * (self.latent_width // self.patch_size[2])

    @property
    def num_chunks(self) -> int:
        return self.chunk_specs.shape[0]

    def segment_index(self, chunk_index: int) -> int:
        return int(torch.searchsorted(self.segment_starts, torch.tensor(chunk_index), right=True).item() - 1)

    def append_motion(self, chunk_index: int, token_tags: torch.Tensor):
        """Add a motion prompt at its chunk boundary; reuse all media geometry."""
        origin, media_origin = self.policy.motion_prompt_origin, self.policy.media_origin
        tags = token_tags.long().cpu()
        if origin + tags.numel() > media_origin:
            raise ValueError("Motion prompts must fit between motion_prompt_origin and media_origin")
        positions = torch.zeros(tags.numel(), 3, dtype=torch.float64)
        positions[:, 0] = origin + torch.arange(tags.numel(), dtype=torch.float64)
        return replace(
            self,
            prompt_position_ids=(*self.prompt_position_ids, positions),
            prompt_token_tags=(*self.prompt_token_tags, tags),
            segment_starts=torch.cat((self.segment_starts, torch.tensor([chunk_index]))),
        )

    def chunk_slices(self, chunk_index: int) -> tuple[slice, slice]:
        frame_start, frame_end, audio_start, audio_end = (int(value) for value in self.chunk_specs[chunk_index].tolist())
        return (
            slice(
                frame_start * self.rows_per_frame,
                frame_end * self.rows_per_frame,
            ),
            slice(audio_start, audio_end),
        )

    def chunk_row_slice(self, chunk_index: int) -> slice:
        video, audio = self.chunk_slices(chunk_index)
        return slice(
            video.start + AUDIO_CHANNELS * audio.start,
            video.stop + AUDIO_CHANNELS * audio.stop,
        )

    def entry_position_ids(self, key: CleanCacheKey) -> torch.Tensor:
        kind, index = key
        if key == GLOBAL_KEY:
            return self.global_position_ids
        if key == REFERENCE_KEY:
            return self.reference_position_ids
        if kind == "prompt":
            return self.prompt_position_ids[index]
        if kind == "chunk":
            return self.target_position_ids[self.chunk_row_slice(index)]
        raise KeyError(f"unknown CleanAR cache entry {key}")

    def chunk_token_tags(self, chunk_index: int) -> torch.Tensor:
        return self.target_token_tags[self.chunk_row_slice(chunk_index)]

    def global_layout(self) -> MiniMaxH3PackedSequence:
        count = self.global_position_ids.shape[0]
        return MiniMaxH3PackedSequence(
            sequence_length=count,
            position_ids=self.global_position_ids,
            token_tags=self.global_token_tags,
            video_indices=torch.empty(0, dtype=torch.long),
            audio_indices=torch.empty(0, dtype=torch.long),
            text_indices=torch.arange(count),
            num_condition_video_rows=0,
            num_condition_audio_rows=0,
        )

    def reference_layout(self) -> MiniMaxH3PackedSequence:
        count = self.reference_position_ids.shape[0]
        video_end = self.num_reference_video_rows
        return MiniMaxH3PackedSequence(
            sequence_length=count,
            position_ids=self.reference_position_ids,
            token_tags=self.reference_token_tags,
            video_indices=torch.arange(video_end),
            audio_indices=torch.arange(video_end, count),
            text_indices=torch.empty(0, dtype=torch.long),
            num_condition_video_rows=video_end,
            num_condition_audio_rows=self.num_reference_audio_rows,
        )

    def prompt_layout(self, prompt_index: int) -> MiniMaxH3PackedSequence:
        positions = self.prompt_position_ids[prompt_index]
        count = positions.shape[0]
        return MiniMaxH3PackedSequence(
            sequence_length=count,
            position_ids=positions,
            token_tags=self.prompt_token_tags[prompt_index],
            video_indices=torch.empty(0, dtype=torch.long),
            audio_indices=torch.empty(0, dtype=torch.long),
            text_indices=torch.arange(count),
            num_condition_video_rows=0,
            num_condition_audio_rows=0,
        )

    def chunk_layout(self, chunk_index: int) -> MiniMaxH3PackedSequence:
        positions = self.chunk_position_ids(chunk_index, chunk_index)
        video_slice, _ = self.chunk_slices(chunk_index)
        video_rows = video_slice.stop - video_slice.start
        count = positions.shape[0]
        return MiniMaxH3PackedSequence(
            sequence_length=count,
            position_ids=positions,
            token_tags=self.chunk_token_tags(chunk_index),
            video_indices=torch.arange(video_rows),
            audio_indices=torch.arange(video_rows, count),
            text_indices=torch.empty(0, dtype=torch.long),
            num_condition_video_rows=0,
            num_condition_audio_rows=0,
        )

    def fixed_keys(self) -> tuple[CleanCacheKey, ...]:
        keys = []
        if self.policy.sink_text_condition:
            keys.append(GLOBAL_KEY)
        if self.policy.sink_ref_condition:
            keys.append(REFERENCE_KEY)
        return tuple(keys)

    def segment_sink_indices(self, segment: int) -> tuple[int, ...]:
        count = self.policy.sink_chunk_num
        segment_start = int(self.segment_starts[segment])
        if segment == 0:
            return tuple(range(count))
        return tuple(range(segment_start - count, segment_start))

    def _segment_rope_window(self, current_chunk_index: int) -> tuple[tuple[int, ...], int, int]:
        """Return current sinks, local origin and current local chunk index."""
        segment = self.segment_index(current_chunk_index)
        sinks = self.segment_sink_indices(segment)
        origin_chunk = sinks[0]
        return sinks, origin_chunk, current_chunk_index - origin_chunk

    def chunk_position_ids(self, chunk_index: int, current_chunk_index: int) -> torch.Tensor:
        """Map one media chunk onto the active segment's bounded RoPE clock."""
        positions = self.entry_position_ids(chunk_key(chunk_index))
        sinks, origin_chunk, current_local_index = self._segment_rope_window(current_chunk_index)
        source_origin = self.entry_position_ids(chunk_key(origin_chunk))[0, 0]
        target_origin = self.entry_position_ids(chunk_key(0))[0, 0]
        segment_shift = target_origin - source_origin

        # A segment starts with its bridge sinks at media slot zero. Before the
        # window fills, one shared shift keeps every media interval continuous.
        if chunk_index in sinks or current_local_index <= self.policy.kv_cache_max_t_rope:
            shift = segment_shift
        else:
            max_slot_chunk = origin_chunk + self.policy.kv_cache_max_t_rope
            max_slot_time = self.entry_position_ids(chunk_key(max_slot_chunk))[0, 0] + segment_shift
            current_time = self.entry_position_ids(chunk_key(current_chunk_index))[0, 0]
            # Shift the complete recent window together so its relative A/V
            # timing is unchanged while the current chunk remains at max slot.
            shift = max_slot_time - current_time

        if not bool(shift):
            return positions
        positions = positions.clone()
        positions[:, 0] += shift
        return positions

    def prefix_position_ids(
        self,
        keys: tuple[CleanCacheKey, ...],
        current_chunk_index: int,
    ) -> torch.Tensor:
        positions = [self.chunk_position_ids(key[1], current_chunk_index) if key[0] == "chunk" else self.entry_position_ids(key) for key in keys]
        if positions:
            return torch.cat(positions)
        return self.global_position_ids.new_empty((0, 3))

    def clean_history_indices(self, chunk_index: int) -> tuple[int, ...]:
        segment = self.segment_index(chunk_index)
        history = {index for index in self.segment_sink_indices(segment) if index < chunk_index}
        history.update(
            range(
                max(0, chunk_index - self.policy.latest_chunk_num + 1),
                chunk_index,
            )
        )
        return tuple(sorted(history))

    def clean_prefix_keys(self, chunk_index: int) -> tuple[CleanCacheKey, ...]:
        return self.fixed_keys() + tuple(chunk_key(index) for index in self.clean_history_indices(chunk_index))

    def noisy_prefix_keys(self, chunk_index: int) -> tuple[CleanCacheKey, ...]:
        segment = self.segment_index(chunk_index)
        return self.fixed_keys() + (prompt_key(segment),) + tuple(chunk_key(index) for index in self.clean_history_indices(chunk_index))


def build_clean_streaming_plan_from_geometry(
    *,
    text_token_tags: tuple[torch.Tensor, ...],
    references: list[MiniMaxH3PreparedReference],
    num_reference_audio_latents: int,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    segment_starts: tuple[int, ...],
    policy: CleanAttentionPolicy,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> MiniMaxH3CleanStreamingPlan:
    """Build metadata from the scheduler's validated request geometry."""
    _, patch_h, patch_w = patch_size
    chunk_specs = build_chunk_specs(num_latent_frames, num_audio_latents, policy)

    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    global_tags = text_token_tags[0].long().cpu()
    global_count = global_tags.numel()
    global_positions = torch.zeros(global_count, 3, dtype=torch.float64)
    global_positions[:, 0] = torch.arange(global_count, dtype=torch.float64)

    reference_positions = []
    reference_tags = []
    num_reference_video_rows = 0
    for ref_index, reference in enumerate(references):
        rows = reference.num_video_rows
        spatial, _ = _frame_position_grid(
            reference.latent_height,
            reference.latent_width,
            patch_h,
            patch_w,
        )
        positions = torch.zeros(rows, 3, dtype=torch.float64)
        positions[:, 0] = global_count + ref_index
        positions[:, 1:] = spatial
        reference_positions.append(positions)
        reference_tags.append(torch.full((rows,), VIDEO_TAG, dtype=torch.long))
        num_reference_video_rows += rows

    num_reference_audio_rows = num_reference_audio_latents * AUDIO_CHANNELS
    if num_reference_audio_latents:
        audio_positions = torch.zeros(num_reference_audio_rows, 3, dtype=torch.float64)
        times = global_count + len(references) + torch.arange(num_reference_audio_latents, dtype=torch.float64)
        audio_positions[:, 0] = times.repeat(AUDIO_CHANNELS)
        audio_positions[:, 2] = width_grid[[0, -1]].repeat_interleave(num_reference_audio_latents)
        reference_positions.append(audio_positions)
        reference_tags.append(
            torch.full(
                (num_reference_audio_rows,),
                AUDIO_TAG,
                dtype=torch.long,
            )
        )
    reference_position_ids = torch.cat(reference_positions)
    reference_token_tags = torch.cat(reference_tags)

    ref_end_time = global_count + len(references) + num_reference_audio_latents
    prompt_origin = policy.motion_prompt_origin
    prompt_lengths = [int(tags.numel()) for tags in text_token_tags[1:]]
    media_origin = policy.media_origin
    if prompt_origin < ref_end_time or media_origin < prompt_origin + max(prompt_lengths):
        raise ValueError("CleanAR RoPE origins overlap preceding conditions")
    prompt_positions = tuple(
        torch.stack(
            (
                prompt_origin + torch.arange(tags.numel(), dtype=torch.float64),
                torch.zeros(tags.numel(), dtype=torch.float64),
                torch.zeros(tags.numel(), dtype=torch.float64),
            ),
            dim=1,
        )
        for tags in text_token_tags[1:]
    )
    prompt_tags = tuple(tags.long().cpu() for tags in text_token_tags[1:])

    frame_times = _temporal_position_grid(num_latent_frames, float(media_origin))
    target_positions = []
    target_tags = []
    for frame_start, frame_end, audio_start, audio_end in chunk_specs:
        video_positions = torch.zeros(
            (frame_end - frame_start) * frame_grid.shape[0],
            3,
            dtype=torch.float64,
        )
        video_positions[:, 0] = frame_times[frame_start:frame_end].repeat_interleave(frame_grid.shape[0])
        video_positions[:, 1:] = frame_grid.repeat(frame_end - frame_start, 1)
        audio_count = audio_end - audio_start
        audio_positions = torch.zeros(
            AUDIO_CHANNELS * audio_count,
            3,
            dtype=torch.float64,
        )
        audio_positions[:, 0] = (media_origin + torch.arange(audio_start, audio_end, dtype=torch.float64)).repeat(AUDIO_CHANNELS)
        audio_positions[:, 2] = width_grid[[0, -1]].repeat_interleave(audio_count)
        target_positions.extend((video_positions, audio_positions))
        target_tags.extend(
            (
                torch.full(
                    (video_positions.shape[0],),
                    VIDEO_TAG,
                    dtype=torch.long,
                ),
                torch.full(
                    (audio_positions.shape[0],),
                    AUDIO_TAG,
                    dtype=torch.long,
                ),
            )
        )

    return MiniMaxH3CleanStreamingPlan(
        global_position_ids=global_positions,
        global_token_tags=global_tags,
        reference_position_ids=reference_position_ids,
        reference_token_tags=reference_token_tags,
        num_reference_video_rows=num_reference_video_rows,
        num_reference_audio_rows=num_reference_audio_rows,
        prompt_position_ids=prompt_positions,
        prompt_token_tags=prompt_tags,
        target_position_ids=torch.cat(target_positions),
        target_token_tags=torch.cat(target_tags),
        chunk_specs=torch.tensor(chunk_specs, dtype=torch.long),
        segment_starts=torch.tensor(segment_starts, dtype=torch.long),
        num_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=num_audio_latents,
        policy=policy,
        patch_size=patch_size,
    )
