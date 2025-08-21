from dataclasses import dataclass
from typing import Any, List, Optional

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
    audio_dit_blocks: List[Any] = None
    valid_patch_length: Optional[int] = None
    hints: List[Any] = None
    context_scale: float = 1.0
