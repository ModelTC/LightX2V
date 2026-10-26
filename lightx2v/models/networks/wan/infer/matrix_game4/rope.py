from typing import NamedTuple

import torch
import triton

from lightx2v.common.kvcache.attn.sage import QuantizedRingAttentionState, SageRingAttentionState, encode_sage_current
from lightx2v.common.kvcache.rolling import RingKVCacheView
from lightx2v.common.ops.rope.kernels.causal_rope import causal_rope_kernel, update_sink_key_kernel


def apply_dynamic_causal_rope(query, key, grid_sizes, freqs, start_frame):
    return _dynamic_causal_rope(query, key, grid_sizes, freqs, start_frame)


@torch.library.custom_op("lightx2v::matrix_game4_causal_rope", mutates_args=(), device_types="cuda")
def _dynamic_causal_rope(query: torch.Tensor, key: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor, start_frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    query, key = query.contiguous(), key.contiguous()
    query_out, key_out = torch.empty_like(query), torch.empty_like(key)
    pairs = key.numel() // 2
    causal_rope_kernel[(triton.cdiv(pairs, 256),)](
        query,
        key,
        query_out,
        key_out,
        torch.view_as_real(freqs),
        grid_sizes,
        start_frame,
        pairs,
        key.shape[1],
        key.shape[2],
        True,
        256,
        enable_fp_fusion=False,
        num_warps=4,
    )
    return query_out, key_out


@_dynamic_causal_rope.register_fake
def _dynamic_causal_rope_fake(query, key, grid_sizes, freqs, start_frame):
    return torch.empty_like(query), torch.empty_like(key)


def apply_dynamic_causal_rope_single(key, grid_sizes, freqs, start_frame):
    return _dynamic_causal_rope_single(key, grid_sizes, freqs, start_frame)


@torch.library.custom_op("lightx2v::matrix_game4_causal_rope_single", mutates_args=(), device_types="cuda")
def _dynamic_causal_rope_single(key: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor, start_frame: torch.Tensor) -> torch.Tensor:
    key = key.contiguous()
    output = torch.empty_like(key)
    pairs = key.numel() // 2
    causal_rope_kernel[(triton.cdiv(pairs, 256),)](
        key,
        key,
        output,
        output,
        torch.view_as_real(freqs),
        grid_sizes,
        start_frame,
        pairs,
        key.shape[1],
        key.shape[2],
        False,
        256,
        enable_fp_fusion=False,
        num_warps=4,
    )
    return output


@_dynamic_causal_rope_single.register_fake
def _dynamic_causal_rope_single_fake(key, grid_sizes, freqs, start_frame):
    return torch.empty_like(key)


class SinkRoPEState(NamedTuple):
    raw_key: torch.Tensor
    valid: torch.Tensor | None
    window_frames: int
    segment_tokens: int
    rotated_key: torch.Tensor | None = None


class MatrixGame4AttentionState(NamedTuple):
    ring: RingKVCacheView | SageRingAttentionState | QuantizedRingAttentionState
    sink: SinkRoPEState

    @property
    def bindings(self):
        scratch = () if self.sink.rotated_key is None else (self.sink.rotated_key,)
        return *self.ring.bindings, self.sink.raw_key, *scratch

    @property
    def graph_key(self):
        return getattr(self.ring, "graph_key", None)


class SinkRoPE:
    """Keep the raw sink key needed by a moving relative RoPE anchor."""

    def __init__(self, config):
        self.sink_frames = config["ar_config"]["sink_size"]
        self.fixed_sink = config.get("sink_rope_mode", "fixed") == "fixed"
        self.quantized = config["self_attn_1_type"] == "sage_attn2"
        self._sink_keys = {}

    def prepare(self, layer_id, ring, sink_tokens):
        encoded = isinstance(ring, QuantizedRingAttentionState)
        prototype = ring.quantized.key_mean if encoded else ring.key
        shape = (sink_tokens, *prototype.shape[-2:])
        signature = prototype.data_ptr(), prototype.shape, prototype.stride(), sink_tokens
        cached = self._sink_keys.get(layer_id)
        if cached is None or cached[0] != signature:
            cached = signature, prototype.new_zeros(shape), prototype.new_empty(shape) if encoded else None
            self._sink_keys[layer_id] = cached
        quantized = ring.quantized if self.quantized else None
        segment_tokens = quantized.key_int8.shape[1] if self.quantized else 1
        frame_tokens = sink_tokens // self.sink_frames
        capacity = ring.capacity if encoded else ring.key.shape[0]
        return SinkRoPEState(cached[1], quantized.key_valid if self.quantized else None, capacity // frame_tokens, segment_tokens, cached[2])

    def update(self, raw_key, state, ring, grid_sizes, freqs, start_frame):
        encoded = isinstance(ring, QuantizedRingAttentionState)
        key_ring = state.rotated_key if encoded else ring.key
        _update_sink_key(
            raw_key,
            state.raw_key,
            key_ring,
            state.valid,
            grid_sizes,
            torch.view_as_real(freqs),
            start_frame,
            state.window_frames,
            state.segment_tokens,
            self.fixed_sink,
            self.quantized,
        )
        if encoded and self.fixed_sink:
            # Only the small sink's RoPE anchor changes; recent encoded K/V
            # stay resident and untouched. The device condition matches the
            # source kernel, so graph replay does not read uninitialized scratch.
            encode_sage_current(
                state.rotated_key,
                state.rotated_key[:0],
                ring.slot_metadata,
                ring.quantized,
                ring.capacity,
                cache_values=False,
                physical_start=0,
                position=start_frame,
                enabled_after=state.window_frames,
            )

    def reset(self, keep_storage=False):
        if keep_storage:
            for _, raw_key, _ in self._sink_keys.values():
                raw_key.zero_()
        else:
            self._sink_keys.clear()


@torch.library.custom_op("lightx2v::matrix_game4_update_sink_key", mutates_args=("raw_sink_key", "key_ring", "valid"), device_types="cuda")
def _update_sink_key(
    raw_key: torch.Tensor,
    raw_sink_key: torch.Tensor,
    key_ring: torch.Tensor,
    valid: torch.Tensor | None,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: torch.Tensor,
    window_frames: int,
    segment_tokens: int,
    fixed_sink: bool,
    quantized: bool,
) -> None:
    pairs = raw_sink_key.numel() // 2
    update_sink_key_kernel[(triton.cdiv(pairs, 256),)](
        raw_key,
        raw_sink_key,
        key_ring,
        valid,
        freqs,
        grid_sizes,
        start_frame,
        pairs,
        raw_key.shape[1],
        raw_key.shape[2],
        window_frames,
        triton.cdiv(raw_sink_key.shape[0], segment_tokens),
        fixed_sink,
        quantized,
        256,
        enable_fp_fusion=False,
        num_warps=4,
    )


@_update_sink_key.register_fake
def _update_sink_key_fake(raw_key, raw_sink_key, key_ring, valid, grid_sizes, freqs, start_frame, window_frames, segment_tokens, fixed_sink, quantized):
    return None
