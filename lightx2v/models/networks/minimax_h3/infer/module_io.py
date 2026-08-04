from dataclasses import dataclass

import torch


@dataclass
class MiniMaxH3PreInferOutput:
    hidden_states: torch.Tensor
    temb: torch.Tensor
    timestep_indices: torch.Tensor
    adaln_indices: torch.Tensor
    rotary_emb: tuple[torch.Tensor, torch.Tensor]
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor


@dataclass
class MiniMaxH3VelocityOutput:
    video: torch.Tensor
    audio: torch.Tensor
