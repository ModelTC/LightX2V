from dataclasses import dataclass

import torch


@dataclass
class NeoppPreInferModuleOutput:
    image_embeds: torch.Tensor
    t: torch.Tensor
    z: torch.Tensor
    image_token_num: int
    timestep_embeddings: torch.Tensor
