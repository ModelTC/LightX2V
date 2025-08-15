from dataclasses import dataclass
from typing import List

import torch


@dataclass
class WanPreInferModuleOutput:
    embed: torch.Tensor
    grid_sizes: torch.Tensor
    x: torch.Tensor
    embed0: torch.Tensor
    seq_lens: torch.Tensor
    freqs: torch.Tensor
    context: torch.Tensor
    audio_dit_blocks: List = None
    valid_patch_length: int = None
