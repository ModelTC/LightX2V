"""Module I/O definitions for LTX2 infer classes."""

from dataclasses import dataclass

import torch


@dataclass
class TransformerArgs:
    """Transformer arguments matching source TransformerArgs structure."""

    x: torch.Tensor | None = None
    context: torch.Tensor | None = None
    context_mask: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    embedded_timestep: torch.Tensor | None = None
    positional_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None
    cross_positional_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None
    cross_scale_shift_timestep: torch.Tensor | None = None
    cross_gate_timestep: torch.Tensor | None = None


@dataclass
class LTX2PreInferModuleOutput:
    """Output from LTX2PreInfer module."""

    video_args: TransformerArgs | None = None
    audio_args: TransformerArgs | None = None
