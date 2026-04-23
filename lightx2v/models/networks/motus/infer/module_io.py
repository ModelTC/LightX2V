from dataclasses import dataclass, field
from typing import Any

import torch

from lightx2v.models.networks.wan.infer.module_io import GridOutput, WanPreInferModuleOutput


@dataclass(kw_only=True)
class MotusPreInferModuleOutput(WanPreInferModuleOutput):
    state: torch.Tensor
    first_frame: torch.Tensor
    instruction: str
    t5_embeddings: list[torch.Tensor]
    vlm_inputs: list[dict[str, Any]]
    image_context: torch.Tensor | None
    und_tokens: torch.Tensor
    condition_frame_latent: torch.Tensor
    adapter_args: dict[str, Any] = field(default_factory=dict)
    conditional_dict: dict[str, Any] = field(default_factory=dict)
