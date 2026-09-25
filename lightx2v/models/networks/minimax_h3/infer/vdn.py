"""H3 layout, frame reductions and sequence exchange for VDN attention."""

import torch
import torch.distributed as dist

from lightx2v.common.ops.attn.ulysses_a2a import TorchUlyssesA2A
from lightx2v.common.ops.attn.vdn_window_attn import VDNLayout


def get_layout(scheduler, config):
    source = scheduler.layout_cpu
    indices = source.video_indices[source.num_condition_video_rows :]
    patch = tuple(scheduler.config.get("patch_size", (1, 2, 2)))
    window = config["softmax_attention"]
    layout = VDNLayout(
        sequence_length=source.sequence_length,
        video_start=int(indices[0]),
        num_frames=scheduler.num_latent_frames,
        frame_size=(scheduler.latent_height // patch[1], scheduler.latent_width // patch[2]),
        text_length=source.text_indices.numel(),
        chunk=int(window["chunk"]),
        radius=int(window["radius"]),
    )
    if layout.video_end != layout.sequence_length or not torch.equal(indices, torch.arange(layout.video_start, layout.video_end)):
        raise ValueError("VDN generated video must be a contiguous frame-major sequence tail")
    if not torch.equal(source.text_indices, torch.arange(layout.text_length)):
        raise ValueError("VDN Qwen conditioning must be a contiguous sequence prefix")
    return layout


def frame_mean(x, layout, state, group):
    spatial = layout.tokens_per_frame
    if state is None:
        return x[layout.video_start : layout.video_end].reshape(layout.num_frames, spatial, -1).mean(1, dtype=torch.float32)

    shard_start = state.aux_length + dist.get_rank(group) * state.main_shard_length
    start = max(shard_start, layout.video_start)
    end = min(shard_start + state.main_shard_length, layout.video_end)
    sums = x.new_zeros((layout.num_frames, x.shape[-1]), dtype=torch.float32)
    if start < end:
        local_start = state.aux_length + start - shard_start
        rows = x[local_start : local_start + end - start]
        frame, offset = divmod(start - layout.video_start, spatial)
        cursor = 0
        if offset:
            cursor = min(spatial - offset, rows.shape[0])
            sums[frame] = rows[:cursor].sum(0, dtype=torch.float32)
            frame += 1
        full_frames = (rows.shape[0] - cursor) // spatial
        if full_frames:
            stop = cursor + full_frames * spatial
            sums[frame : frame + full_frames] = rows[cursor:stop].reshape(full_frames, spatial, x.shape[-1]).sum(1, dtype=torch.float32)
            cursor = stop
            frame += full_frames
        if cursor < rows.shape[0]:
            sums[frame] = rows[cursor:].sum(0, dtype=torch.float32)
    # Every SP rank participates, including ranks containing only conditioning.
    dist.all_reduce(sums, group=group)
    return sums / spatial


def sequence_to_heads(packed, aux_length, group):
    """Exchange sequence shards, keeping the replicated prefix only once."""
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    local, heads, width = packed[aux_length:].shape
    head_count = heads // world
    send = packed[aux_length:].reshape(local, world, head_count, width).permute(1, 0, 2, 3).contiguous()
    received, _ = TorchUlyssesA2A.exchange(send, group)
    main = received.reshape(world * local, head_count, width)
    if not aux_length:
        return main
    aux = packed[:aux_length, rank * head_count : (rank + 1) * head_count]
    return torch.cat((aux, main), dim=0)


def heads_to_sequence(output, aux_length, group):
    world = dist.get_world_size(group)
    length, heads, dim = output[aux_length:].shape
    local = length // world
    send = output[aux_length:].reshape(world, local, heads, dim).contiguous()
    received, _ = TorchUlyssesA2A.exchange(send, group)
    main = received.permute(1, 0, 2, 3).reshape(local, world * heads, dim)
    # Linear attention has no output on the replicated conditioning prefix.
    return torch.cat((main.new_zeros((aux_length, world * heads, dim)), main), dim=0) if aux_length else main


def video_rows(length, layout, state, group, device):
    positions = torch.arange(length, device=device)
    if state is not None:
        positions[state.aux_length :] += dist.get_rank(group) * state.main_shard_length
    # Generated boundary frames, input keyframes and Qwen vision rows do not
    # receive a linear contribution.
    return (positions >= layout.video_start + layout.tokens_per_frame) & (positions < layout.video_end - layout.tokens_per_frame)
