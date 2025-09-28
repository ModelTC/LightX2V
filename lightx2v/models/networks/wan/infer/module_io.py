from dataclasses import dataclass, field
from typing import Any, Dict

import torch


@dataclass
class GridOutput:
    tensor: torch.Tensor
    tuple: tuple


@dataclass
class WanPreInferModuleOutput:
    embed: torch.Tensor
    grid_sizes: GridOutput
    x: torch.Tensor
    embed0: torch.Tensor
    seq_lens: torch.Tensor
    freqs: torch.Tensor
    context: torch.Tensor
    adapter_args: Dict[str, Any] = field(default_factory=dict)
