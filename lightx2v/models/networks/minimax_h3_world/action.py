"""Native H3-World action text, rotary positions, and directed attention mask."""

from dataclasses import dataclass, fields

import numpy as np
import torch
from torch.nn.attention.flex_attention import create_block_mask

from lightx2v.models.networks.minimax_h3.packing import MiniMaxH3PackedSequence, _temporal_position_grid, video_latent_num_frames

ACTION_KEYS = ("W", "A", "S", "D", "I", "J", "K", "L", "F")
_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_MASK_BLOCK_SIZE = 128


@dataclass(frozen=True)
class MiniMaxH3WorldPackedSequence(MiniMaxH3PackedSequence):
    text_segment_lengths: tuple[int, ...] = ()
    action_block_mask: object = None


def _format_action_texts(actions: np.ndarray) -> list[str]:
    """The released nine-bit formatter, including cancellation of opposite keys."""
    sentences = []
    for row in actions:
        on = {key: bool(value > 0) for key, value in zip(ACTION_KEYS, row)}
        for first, second in (("W", "S"), ("A", "D"), ("J", "L"), ("I", "K")):
            if on[first] and on[second]:
                on[first] = on[second] = False
        motion = [word for key, word in (("W", "walks forward"), ("S", "walks backward"), ("A", "strafes left"), ("D", "strafes right")) if on[key]]
        camera = []
        for key, side in (("J", "left"), ("L", "right")):
            if on[key]:
                camera.append(f"pans {side} {'sharply' if on['F'] else 'slowly'}")
        for key, word in (("I", "tilts down"), ("K", "tilts up")):
            if on[key]:
                camera.append(word)
        motion_clause = " and ".join(motion) if motion else "stands still"
        camera_clause = " and ".join(camera) if camera else ("follows him" if motion else "holds steady")
        sentences.append(f"the man {motion_clause}, camera {camera_clause}")
    return sentences


def _key_vector(keys) -> np.ndarray:
    if isinstance(keys, str):
        keys = keys.replace(",", " ").replace("+", " ").split()
    if isinstance(keys, dict):
        unknown = set(keys) - set(ACTION_KEYS)
        enabled = [key for key, value in keys.items() if value]
    elif isinstance(keys, list):
        if not all(isinstance(key, str) for key in keys):
            raise ValueError("Action keys must be names from W,A,S,D,I,J,K,L,F")
        unknown = set(keys) - set(ACTION_KEYS)
        enabled = keys
    else:
        raise ValueError("Action keys must be a list, a name-to-value object, or a string")
    if unknown:
        raise ValueError(f"Unknown H3-World action keys: {sorted(unknown)}; expected {ACTION_KEYS}")
    return np.array([key in enabled for key in ACTION_KEYS], dtype=np.float32)


def _frame_index(value, num_frames: int, *, endpoint: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"Action frame indices must be integers, got {value!r}")
    try:
        index = int(value)
    except ValueError as error:
        raise ValueError(f"Action frame indices must be integers, got {value!r}") from error
    limit = num_frames if endpoint else num_frames - 1
    if not 0 <= index <= limit:
        raise ValueError(f"Action frame index {index} is outside [0, {limit}]")
    return index


def action_segments_to_texts(world_segments: list[dict], num_frames: int) -> list[str]:
    """Convert config world_segments to one action sentence per latent frame.

    Frame indices use the output 24 fps timeline and end_frame is exclusive.
    Omitted frames have no keys pressed; overlapping segments union their keys.
    Keys are pooled over repeating (1,4,4,4,4) frame spans. An empty list means
    no keys pressed throughout the video; F (fast pan) is supplied explicitly.
    """
    num_latents = video_latent_num_frames(num_frames)
    if num_latents <= 0:
        raise ValueError("H3-World requires a positive aligned frame count")
    if not isinstance(world_segments, list):
        raise ValueError("H3-World requires world_segments to be a list in the config")
    actions = np.zeros((num_frames, len(ACTION_KEYS)), dtype=np.float32)
    for entry in world_segments:
        if not isinstance(entry, dict) or not {"start_frame", "end_frame", "keys"} <= entry.keys():
            raise ValueError("Each world_segments entry requires start_frame, end_frame, and keys")
        start = _frame_index(entry["start_frame"], num_frames)
        stop = _frame_index(entry["end_frame"], num_frames, endpoint=True)
        if stop <= start:
            raise ValueError("Action end_frame must be greater than start_frame (end is exclusive)")
        actions[start:stop] = np.maximum(actions[start:stop], _key_vector(entry["keys"]))
    spans = [_FRAMES_PER_LATENT[index % len(_FRAMES_PER_LATENT)] for index in range(num_latents)]
    starts = np.cumsum([0] + spans[:-1])
    actions = np.stack([actions[start : start + span].max(axis=0) for start, span in zip(starts, spans)])
    return _format_action_texts(actions)


def action_mask_rule(q, kv, annotation_ids: torch.Tensor, frame_ids: torch.Tensor) -> torch.Tensor:
    """Directed binding: A_k keys are visible only to A_k and video frame k.

    A_k queries can read static text, conditions, audio, A_k, and video frame k.
    All pairs of non-action rows remain visible. IDs use -1 for other rows;
    callers handle any padding outside the real sequence separately.
    """
    action_q, action_kv = annotation_ids[q], annotation_ids[kv]
    frame_q, frame_kv = frame_ids[q], frame_ids[kv]
    same_action = (action_q >= 0) & (action_kv >= 0) & (action_q == action_kv)
    video_reads_own_action = (frame_q >= 0) & (action_kv >= 0) & (frame_q == action_kv)
    leak_out = (action_kv >= 0) & ~same_action & ~video_reads_own_action
    leak_in = (action_q >= 0) & (frame_kv >= 0) & (action_q != frame_kv)
    return ~(leak_out | leak_in)


def prepare_h3_world_layout(
    layout: MiniMaxH3PackedSequence,
    num_head_tokens: int,
    action_token_lengths: list[int],
    num_latent_frames: int,
    rows_per_frame: int,
) -> MiniMaxH3WorldPackedSequence:
    """Copy a device layout, then prepare action RoPE and reusable attention data."""
    if num_head_tokens <= 0 or rows_per_frame <= 0 or num_latent_frames <= 0:
        raise ValueError("H3-World head length, frame count, and rows per frame must be positive")
    if len(action_token_lengths) != num_latent_frames or any(length <= 0 for length in action_token_lengths):
        raise ValueError("H3-World needs one nonempty action text segment per latent frame")
    num_text_tokens = num_head_tokens + sum(action_token_lengths)
    if layout.text_indices.numel() != num_text_tokens:
        raise ValueError("H3-World head and action lengths do not match the packed text rows")
    num_video_rows = num_latent_frames * rows_per_frame
    if layout.video_indices.numel() - layout.num_condition_video_rows != num_video_rows:
        raise ValueError("H3-World target video rows do not match the latent frame geometry")

    offsets = _temporal_position_grid(num_latent_frames, 0.0)
    origin = float(num_text_tokens) - float(offsets[-1]) - 1.0
    if origin < num_head_tokens:
        raise ValueError("The H3-World action rotary span overlaps the head; increase action text token lengths")
    device = layout.position_ids.device
    position_ids = layout.position_ids.clone()
    offsets = offsets.to(device)
    sequence_length = layout.sequence_length
    # FlexAttention evaluates edge blocks at rounded-up indices as well.
    # Pad both lookups, then explicitly exclude those rows in mask_mod.
    padded_length = ((sequence_length + _MASK_BLOCK_SIZE - 1) // _MASK_BLOCK_SIZE) * _MASK_BLOCK_SIZE
    annotation_ids = torch.full((padded_length,), -1, dtype=torch.int32, device=device)
    frame_ids = torch.full_like(annotation_ids, -1)
    cursor = num_head_tokens
    for frame, length in enumerate(action_token_lengths):
        rows = layout.text_indices[cursor : cursor + length]
        position_ids[rows, 0] = origin + offsets[frame]
        position_ids[rows, 1:] = 0
        annotation_ids[rows] = frame
        cursor += length
    target_video_rows = layout.video_indices[layout.num_condition_video_rows :]
    frame_ids[target_video_rows] = torch.arange(num_latent_frames, dtype=torch.int32, device=device).repeat_interleave(rows_per_frame)

    def mask_mod(batch, head, q, kv):
        return (q < sequence_length) & (kv < sequence_length) & action_mask_rule(q, kv, annotation_ids, frame_ids)

    block_mask = create_block_mask(
        mask_mod,
        None,
        None,
        sequence_length,
        sequence_length,
        device=device,
        BLOCK_SIZE=_MASK_BLOCK_SIZE,
        # This runs once per request. Eager construction avoids a multi-minute
        # reduction-kernel compile on long sequences (PyTorch 2.11).
        _compile=False,
    )
    values = {field.name: getattr(layout, field.name) for field in fields(MiniMaxH3PackedSequence)}
    values["position_ids"] = position_ids
    return MiniMaxH3WorldPackedSequence(
        **values,
        text_segment_lengths=(num_head_tokens, *action_token_lengths),
        action_block_mask=block_mask,
    )
