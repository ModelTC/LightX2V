from dataclasses import dataclass
from typing import Any

import torch

from lightx2v.models.networks.wan.infer.module_io import GridOutput


@dataclass
class MotusPreInferModuleOutput:
    first_frame: torch.Tensor
    state: torch.Tensor
    instruction: str
    t5_embeddings: list[torch.Tensor]
    vlm_inputs: list[dict[str, Any]]
    processed_t5_context: torch.Tensor
    image_context: torch.Tensor | None
    und_tokens: torch.Tensor
    condition_frame_latent: torch.Tensor
    grid_sizes: GridOutput


@dataclass
class MotusPostInferModuleOutput:
    pred_frames: torch.Tensor
    pred_actions: torch.Tensor
