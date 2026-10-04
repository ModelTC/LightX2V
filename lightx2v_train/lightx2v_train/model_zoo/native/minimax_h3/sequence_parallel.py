"""Ulysses sequence parallelism for Diffusers' MiniMax-H3 transformer.

The small text token refiner remains replicated. The packed AV/text sequence
is sharded only across the 50-layer main stack, where uneven Ulysses
all-to-all avoids sequence padding and an SxS attention mask.
"""

from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist

from lightx2v_train.runtime.distributed import (
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    is_sequence_parallel_enabled,
)
from lightx2v_train.runtime.sequence_parallel import (
    all_gather_variable_sequence,
    all_to_all_4d_variable,
    balanced_sequence_lengths,
    balanced_sequence_slice,
)


@dataclass(frozen=True)
class MiniMaxH3SequenceParallelInfo:
    """Per-forward shard metadata passed through checkpointed H3 blocks."""

    sequence_lengths: tuple[int, ...]


def _diffusers_h3_symbols():
    # Keep these imports lazy so the model registry remains importable without
    # the MiniMax-H3 Diffusers build.
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MINIMAX_H3_MODALITY_NUM,
        MiniMaxH3TransformerOutput,
        _apply_rotary_emb,
    )

    return dispatch_attention_fn, MINIMAX_H3_MODALITY_NUM, MiniMaxH3TransformerOutput, _apply_rotary_emb


class MiniMaxH3SequenceParallelAttnProcessor:
    """Full H3 self-attention with uneven Ulysses sequence parallelism."""

    def __init__(self, previous_processor):
        self._attention_backend = getattr(previous_processor, "_attention_backend", None)
        self._parallel_config = getattr(previous_processor, "_parallel_config", None)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: MiniMaxH3SequenceParallelInfo | None = None,
    ) -> torch.Tensor:
        if not isinstance(attention_mask, MiniMaxH3SequenceParallelInfo):
            raise RuntimeError("MiniMax-H3 SP attention requires internal shard metadata. Install it through install_minimax_h3_sequence_parallel().")
        sequence_lengths = attention_mask.sequence_lengths

        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if rotary_emb is not None:
            _, _, _, apply_rotary_emb = _diffusers_h3_symbols()
            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        query = all_to_all_4d_variable(query, scatter_dim=2, gather_dim=1, sequence_lengths=sequence_lengths)
        key = all_to_all_4d_variable(key, scatter_dim=2, gather_dim=1, sequence_lengths=sequence_lengths)
        value = all_to_all_4d_variable(value, scatter_dim=2, gather_dim=1, sequence_lengths=sequence_lengths)

        dispatch_attention_fn, _, _, _ = _diffusers_h3_symbols()
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = all_to_all_4d_variable(hidden_states, scatter_dim=1, gather_dim=2, sequence_lengths=sequence_lengths)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def _local_modality_rows(indices: torch.Tensor, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
    selected = torch.nonzero((indices >= start) & (indices < end), as_tuple=False).flatten()
    local_indices = indices.index_select(0, selected) - start
    return selected, local_indices


def _gather_output_lengths(video_length: int, audio_length: int, device: torch.device):
    local = torch.tensor((video_length, audio_length), device=device, dtype=torch.int64)
    gathered = [torch.empty_like(local) for _ in range(get_sequence_parallel_world_size())]
    dist.all_gather(gathered, local, group=get_sequence_parallel_group())
    video_lengths = tuple(int(value[0].item()) for value in gathered)
    audio_lengths = tuple(int(value[1].item()) for value in gathered)
    return video_lengths, audio_lengths


def _validate_packed_inputs(
    hidden_states,
    audio_hidden_states,
    encoder_hidden_states,
    timestep_indices,
    token_tags,
    position_ids,
    video_indices,
    audio_indices,
    text_indices,
):
    if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
        raise ValueError(f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}.")
    sequence_length = position_ids.shape[0]
    if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
        raise ValueError(f"`token_tags` and `timestep_indices` must match `position_ids`; got {list(token_tags.shape)} and {list(timestep_indices.shape)} for seq_len={sequence_length}.")
    if sequence_length < get_sequence_parallel_world_size():
        raise ValueError(f"MiniMax-H3 packed seq_len={sequence_length} must be at least SP size {get_sequence_parallel_world_size()}.")
    if bool((token_tags < 0).any()):
        raise ValueError("MiniMax-H3 SP requires a padless packed sequence; padding would require an SxS attention mask and disable unmasked FlashAttention.")
    expected = (
        ("video_indices", video_indices, hidden_states.shape[1]),
        ("audio_indices", audio_indices, audio_hidden_states.shape[1]),
        ("text_indices", text_indices, encoder_hidden_states.shape[1]),
    )
    for name, indices, rows in expected:
        if indices.ndim != 1 or indices.numel() != rows:
            raise ValueError(f"{name} must contain one packed index per input row; got {tuple(indices.shape)} for {rows} rows.")
        if indices.device != position_ids.device:
            raise ValueError(f"{name} and position_ids must be on the same device, got {indices.device} and {position_ids.device}.")
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must use an integer index dtype, got {indices.dtype}.")
        if indices.numel():
            if bool(((indices < 0) | (indices >= sequence_length)).any()):
                raise ValueError(f"{name} contains an index outside [0, {sequence_length}).")
            # Rank-order output gathering is correct only because each
            # modality tensor follows packed-sequence order.
            if indices.numel() > 1 and bool((indices[1:] <= indices[:-1]).any()):
                raise ValueError(f"{name} must be strictly increasing and unique.")

    coverage = torch.zeros(sequence_length, device=position_ids.device, dtype=torch.int8)
    for _, indices, _ in expected:
        coverage.scatter_add_(0, indices.to(torch.int64), torch.ones_like(indices, dtype=torch.int8))
    if bool((coverage != 1).any()):
        raise ValueError("video_indices, audio_indices, and text_indices must be disjoint and cover every row of the packed sequence exactly once.")


def _minimax_h3_sequence_parallel_forward(
    self,
    hidden_states: torch.Tensor,
    audio_hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    timestep_indices: torch.Tensor,
    token_tags: torch.Tensor,
    position_ids: torch.Tensor,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    text_indices: torch.Tensor,
    attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
):
    # attention_kwargs is consumed by apply_lora_scale on the installed
    # wrapper, matching Diffusers' dense H3 forward.
    del attention_kwargs
    _validate_packed_inputs(
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
    )

    sequence_length = position_ids.shape[0]
    sequence_lengths = balanced_sequence_lengths(sequence_length)
    start, end = balanced_sequence_slice(sequence_length)
    local_sequence_length = end - start
    sp_info = MiniMaxH3SequenceParallelInfo(sequence_lengths)

    video_rows, local_video_indices = _local_modality_rows(video_indices, start, end)
    audio_rows, local_audio_indices = _local_modality_rows(audio_indices, start, end)
    text_rows, local_text_indices = _local_modality_rows(text_indices, start, end)

    # Keep the two-layer text-only refiner dense and replicated. Sharding it
    # would make each SP rank refine a different attention document.
    text_embeds = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))
    text_embeds = self.token_refiner(text_embeds)
    local_text_embeds = text_embeds.index_select(1, text_rows)

    local_video = hidden_states.index_select(1, video_rows)
    local_audio = audio_hidden_states.index_select(1, audio_rows)
    video_embeds = self.proj_in(local_video.to(self.proj_in.weight.dtype))
    audio_embeds = self.audio_proj_in(local_audio.to(self.audio_proj_in.weight.dtype))

    packed = text_embeds.new_zeros((text_embeds.shape[0], local_sequence_length, text_embeds.shape[-1]))
    packed = packed.index_copy(1, local_text_indices, local_text_embeds)
    packed = packed.index_copy(1, local_video_indices, video_embeds.to(text_embeds.dtype))
    packed = packed.index_copy(1, local_audio_indices, audio_embeds.to(text_embeds.dtype))

    local_position_ids = position_ids.narrow(0, start, local_sequence_length)
    rotary_emb = self.rope(local_position_ids)
    temb = self.time_proj(timestep)
    temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))

    _, modality_num, output_cls, _ = _diffusers_h3_symbols()
    local_timestep_indices = timestep_indices.narrow(0, start, local_sequence_length)
    local_token_tags = token_tags.narrow(0, start, local_sequence_length)
    adaln_indices = local_timestep_indices * modality_num + local_token_tags

    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            packed = self._gradient_checkpointing_func(
                block,
                packed,
                temb,
                adaln_indices,
                rotary_emb,
                sp_info,
            )
        else:
            packed = block(packed, temb, adaln_indices, rotary_emb, sp_info)

    packed = self.norm_out(packed, temb, local_timestep_indices).to(self.proj_out.weight.dtype)
    local_video_output = self.proj_out(packed).index_select(1, local_video_indices)
    local_audio_output = self.audio_proj_out(packed).index_select(1, local_audio_indices)

    video_lengths, audio_lengths = _gather_output_lengths(
        local_video_output.shape[1],
        local_audio_output.shape[1],
        packed.device,
    )
    video_output = all_gather_variable_sequence(local_video_output, video_lengths, dim=1)
    audio_output = all_gather_variable_sequence(local_audio_output, audio_lengths, dim=1)

    if not return_dict:
        return video_output, audio_output
    return output_cls(sample=video_output, audio_sample=audio_output)


_DECORATED_FORWARD = None


def _decorated_forward():
    global _DECORATED_FORWARD
    if _DECORATED_FORWARD is None:
        from diffusers.utils import apply_lora_scale

        _DECORATED_FORWARD = apply_lora_scale("attention_kwargs")(_minimax_h3_sequence_parallel_forward)
    return _DECORATED_FORWARD


def install_minimax_h3_sequence_parallel(transformer):
    """Install H3 SP before FSDP wrapping; no-op when SP is disabled."""

    if not is_sequence_parallel_enabled():
        return transformer
    if getattr(transformer, "_lightx2v_h3_sequence_parallel", False):
        return transformer
    if not hasattr(transformer, "transformer_blocks"):
        raise TypeError("MiniMax-H3 SP requires a Diffusers MiniMaxH3Transformer3DModel.")

    sp_size = get_sequence_parallel_world_size()
    if not transformer.transformer_blocks:
        raise ValueError("MiniMax-H3 SP requires at least one main transformer block.")
    heads = int(transformer.transformer_blocks[0].attn.heads)
    if heads % sp_size != 0:
        raise ValueError(f"MiniMax-H3 num_attention_heads={heads} must be divisible by SP size {sp_size}.")

    for block in transformer.transformer_blocks:
        attn = block.attn
        if int(attn.heads) != heads:
            raise ValueError("All MiniMax-H3 main blocks must use the same attention head count.")
        attn.set_processor(MiniMaxH3SequenceParallelAttnProcessor(attn.processor))

    transformer.forward = MethodType(_decorated_forward(), transformer)
    transformer._lightx2v_h3_sequence_parallel = True
    return transformer


__all__ = [
    "MiniMaxH3SequenceParallelAttnProcessor",
    "MiniMaxH3SequenceParallelInfo",
    "install_minimax_h3_sequence_parallel",
]
