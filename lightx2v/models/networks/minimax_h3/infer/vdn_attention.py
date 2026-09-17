"""VDN hybrid attention over LightX2V H3 weights and sequence-parallel layout.

Window/state semantics follow OpenVDN/vdn-minimax-h3 e02ff077 (Apache-2.0).
The softmax branch reuses LightX2V Ulysses; the state branch exchanges its
additional raw projections with the same TorchUlyssesA2A transport.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from lightx2v.common.ops.attn.ulysses_a2a import TorchUlyssesA2A

from .vdn_linear import linear_readout


@dataclass(frozen=True)
class VDNLayout:
    sequence_length: int
    video_start: int
    num_frames: int
    frame_size: tuple[int, int]
    text_length: int

    @property
    def tokens_per_frame(self):
        return self.frame_size[0] * self.frame_size[1]

    @property
    def video_end(self):
        return self.video_start + self.num_frames * self.tokens_per_frame

    @property
    def bounds(self):
        return tuple(((frame // 5 - 1) * 5, (frame // 5 + 2) * 5 - 1) for frame in range(self.num_frames))

    @property
    def full_cover(self):
        return all(lo <= 0 and hi >= self.num_frames - 1 for lo, hi in self.bounds)


def build_window_block_mask(layout, device, block_size=128):
    """Build exact block sparsity from intervals, without a token-by-token NxN mask."""
    start, end, spatial, frames = layout.video_start, layout.video_end, layout.tokens_per_frame, layout.num_frames

    def mask_mod(batch, head, query, key):
        query_video = (query >= start) & (query < end)
        key_video = (key >= start) & (key < end)
        q_frame = ((query - start) // spatial).clamp(0, frames - 1)
        k_frame = (key - start) // spatial
        lo, hi = (q_frame // 5 - 1) * 5, (q_frame // 5 + 2) * 5 - 1
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
    # Global tokens and both boundary frames are dense rows/columns. For other
    # query rows, the union/intersection of their monotonic windows gives the
    # exact set of key blocks with any/all allowed token pairs.
    normal_start, normal_end = start + spatial, end - spatial
    for row in range(count):
        q_start, q_end = row * block_size, min((row + 1) * block_size, layout.sequence_length)
        first, last = max(q_start, normal_start), min(q_end, normal_end)
        if first >= last:
            any_allowed = torch.ones(count, dtype=torch.bool)
            all_allowed = any_allowed.clone()
        else:
            first_frame, last_frame = (first - start) // spatial, (last - 1 - start) // spatial
            union_lo = start + (first_frame // 5 - 1) * 5 * spatial
            union_hi = start + (last_frame // 5 + 2) * 5 * spatial
            intersect_lo = max(normal_start, min(normal_end, start + (last_frame // 5 - 1) * 5 * spatial))
            intersect_hi = max(normal_start, min(normal_end, start + (first_frame // 5 + 2) * 5 * spatial))
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


class VDNWindowAttention:
    def __init__(self):
        self.backend = "flex_triton"
        self.layout = None
        self.mask = None
        self._mask_key = None
        self._compiled = None
        self.block_stats = None

    def prepare(self, layout, device):
        self.layout = layout
        if device.type != "cuda":
            self.backend = "sdpa_reference"
            return
        self.backend = "flex_triton"
        key = (layout, str(device))
        if key == self._mask_key:
            return
        self.mask, self.block_stats = build_window_block_mask(layout, device)
        self._mask_key = key
        if self._compiled is None:
            self._compiled = torch.compile(flex_attention, dynamic=False, fullgraph=True)

    def apply(self, q, k, v, **kwargs):
        if q.shape[0] != self.layout.sequence_length or k.shape[0] != self.layout.sequence_length:
            raise ValueError("VDN window attention needs the complete packed sequence on each head rank")
        if q.is_cuda:
            out = self._compiled(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], block_mask=self.mask, scale=q.shape[-1] ** -0.5, kernel_options={"BACKEND": "TRITON"})
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


def sequence_to_heads(packed, aux_length, group):
    """Keep the replicated Qwen prefix once; exchange the equal-length main shards."""
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
    # The linear branch only updates target-video rows; the replicated Qwen prefix is zero.
    return torch.cat((main.new_zeros((aux_length, world * heads, dim)), main), dim=0) if aux_length else main


class VDNAttention:
    def __init__(self, config):
        if config.get("tensor_parallel", False):
            raise ValueError("VDN currently supports single-card and sequence parallel inference, not tensor parallel")
        self.num_heads = int(config.get("num_attention_heads", 56))
        self.head_dim = int(config.get("attention_head_dim", 128))
        self.window = VDNWindowAttention()
        self.linear_use_tf32 = bool(config.get("vdn_linear_use_tf32", True))
        self.group = config["device_mesh"].get_group(mesh_dim="seq_p") if config.get("seq_parallel", False) else None
        self._layout_source = None
        self.layout = None

    def prepare_layout(self, scheduler, device):
        source = scheduler.layout_cpu
        if source is not self._layout_source:
            indices = source.video_indices[source.num_condition_video_rows :]
            patch = tuple(scheduler.config.get("patch_size", (1, 2, 2)))
            self.layout = VDNLayout(
                source.sequence_length, int(indices[0]), scheduler.num_latent_frames, (scheduler.latent_height // patch[1], scheduler.latent_width // patch[2]), source.text_indices.numel()
            )
            if self.layout.video_end != self.layout.sequence_length or indices.numel() != self.layout.num_frames * self.layout.tokens_per_frame:
                raise ValueError("VDN target video must occupy a contiguous frame-major sequence tail")
            if not torch.equal(indices, torch.arange(self.layout.video_start, self.layout.video_end)):
                raise ValueError("VDN target video row indices are not contiguous")
            if not torch.equal(source.text_indices, torch.arange(self.layout.text_length)):
                raise ValueError("VDN Qwen rows must form a contiguous prefix")
            self._layout_source = source
        self.window.prepare(self.layout, device)

    def _frame_mean(self, x, state):
        layout = self.layout
        spatial = layout.tokens_per_frame
        if state is None:
            return x[layout.video_start : layout.video_end].reshape(layout.num_frames, spatial, -1).mean(1, dtype=torch.float32)

        rank = dist.get_rank(self.group)
        shard_start = state.aux_length + rank * state.main_shard_length
        shard_end = shard_start + state.main_shard_length
        start = max(shard_start, layout.video_start)
        end = min(shard_end, layout.video_end)
        sums = x.new_zeros((layout.num_frames, x.shape[-1]), dtype=torch.float32)

        if start < end:
            local_start = state.aux_length + start - shard_start
            rows = x[local_start : local_start + end - start]
            frame, offset = divmod(start - layout.video_start, spatial)
            cursor = 0

            # First partial frame, including a shard contained within one frame.
            if offset:
                cursor = min(spatial - offset, rows.shape[0])
                sums[frame] = rows[:cursor].sum(0, dtype=torch.float32)
                frame += 1

            # Each output row has a separate reduction over one complete frame.
            full_frames = (rows.shape[0] - cursor) // spatial
            if full_frames:
                stop = cursor + full_frames * spatial
                sums[frame : frame + full_frames] = rows[cursor:stop].reshape(full_frames, spatial, x.shape[-1]).sum(1, dtype=torch.float32)
                cursor = stop
                frame += full_frames

            # Last partial frame. It is distinct from the first partial frame.
            if cursor < rows.shape[0]:
                sums[frame] = rows[cursor:].sum(0, dtype=torch.float32)

        # Ranks with no video rows still contribute zeros to the same collective.
        dist.all_reduce(sums, group=self.group)
        return sums / spatial

    def apply(self, weights, x, pre, h3_infer):
        self.prepare_layout(h3_infer.scheduler, x.device)
        raw_qkv = h3_infer._project_qkv(weights, x)
        q, k, v = h3_infer._apply_qkv_norm_rope(weights, *raw_qkv, pre.rotary_emb)
        softmax = h3_infer._calculate_attention(weights, q, k, v, pre, self.window)
        del q, k, v
        state = pre.sequence_parallel_state
        branch = weights.vdn
        # Shared H3 projections are retained before normalization and rotary embedding.
        raw_q, raw_k, raw_v = (item.view(-1, self.num_heads, self.head_dim) for item in raw_qkv)
        del raw_qkv
        softmax = softmax.view(-1, self.num_heads, self.head_dim)
        softmax_gate = torch.sigmoid(branch.softmax_gate.apply(x)).view(-1, self.num_heads, 1)
        softmax = softmax * softmax_gate
        result = weights.to_out.apply(softmax.reshape(x.shape[0], -1))
        del softmax, softmax_gate
        if self.layout.full_cover:
            return result
        beta = torch.sigmoid(branch.beta_proj.apply(x))
        gate = torch.sigmoid(branch.output_gate_up.apply(branch.output_gate_down.apply(x))).view(-1, self.num_heads, self.head_dim)
        means = self._frame_mean(x, state)
        head_start = 0
        if state is not None:
            packed = torch.cat((raw_q, raw_k, raw_v, beta[..., None], gate), dim=-1)
            del raw_q, raw_k, raw_v, beta, gate
            packed = sequence_to_heads(packed, state.aux_length, self.group)
            raw_q, raw_k, raw_v, beta, gate = packed.split((self.head_dim, self.head_dim, self.head_dim, 1, self.head_dim), dim=-1)
            beta = beta[..., 0]
            head_start = dist.get_rank(self.group) * (self.num_heads // dist.get_world_size(self.group))
        linear = linear_readout(branch, raw_q, raw_k, raw_v, beta, gate, means, self.layout, head_start, self.linear_use_tf32)
        if state is not None:
            linear = heads_to_sequence(linear, state.aux_length, self.group)
            rank = dist.get_rank(self.group)
            positions = torch.cat(
                (
                    torch.arange(state.aux_length, device=x.device),
                    torch.arange(state.aux_length + rank * state.main_shard_length, state.aux_length + (rank + 1) * state.main_shard_length, device=x.device),
                )
            )
        else:
            positions = torch.arange(x.shape[0], device=x.device)
        video = (positions >= self.layout.video_start) & (positions < self.layout.video_end)
        result[video] += branch.to_out_linear.apply(linear[video].reshape(-1, self.num_heads * self.head_dim))
        return result
