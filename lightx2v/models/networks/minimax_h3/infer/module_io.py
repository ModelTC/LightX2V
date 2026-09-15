from dataclasses import dataclass

import torch


@dataclass
class MiniMaxH3SequenceParallelState:
    """Full-sequence metadata retained while the main DiT is sequence-sharded."""

    aux_length: int
    main_shard_length: int
    timestep_indices: torch.Tensor
    adaln_indices: torch.Tensor
    rotary_emb: tuple[torch.Tensor, torch.Tensor]


@dataclass
class MiniMaxH3PreInferOutput:
    hidden_states: torch.Tensor
    temb: torch.Tensor | None
    timestep_indices: torch.Tensor
    adaln_indices: torch.Tensor
    rotary_emb: tuple[torch.Tensor, torch.Tensor]
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    norm_out_modulation: torch.Tensor | None = None
    sequence_parallel_state: MiniMaxH3SequenceParallelState | None = None


@dataclass
class MiniMaxH3VelocityOutput:
    video: torch.Tensor
    audio: torch.Tensor
