from enum import Enum
from typing import Dict, Literal, NamedTuple, Optional

import torch

_receptive_field_t = Literal["half", "full"]
_inflation_mode_t = Literal["none", "tail", "replicate"]
_memory_device_t = Optional[Literal["cpu", "same"]]
_gradient_checkpointing_t = Optional[Literal["half", "full"]]
_selective_checkpointing_t = Optional[Literal["coarse", "fine"]]


class DiagonalGaussianDistribution:
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor):
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self) -> torch.FloatTensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self) -> torch.Tensor:
        return 0.5 * torch.sum(
            self.mean**2 + self.var - 1.0 - self.logvar,
            dim=list(range(1, self.mean.ndim)),
        )


class MemoryState(Enum):
    """
    State[Disabled]:        No memory bank will be enabled.
    State[Initializing]:    The model is handling the first clip, need to reset the memory bank.
    State[Active]:          There has been some data in the memory bank.
    State[Unset]:           Error state, indicating users didn't pass correct memory state in.
    """

    DISABLED = 0
    INITIALIZING = 1
    ACTIVE = 2
    UNSET = 3


class QuantizerOutput(NamedTuple):
    latent: torch.Tensor
    extra_loss: torch.Tensor
    statistics: Dict[str, torch.Tensor]


class CausalAutoencoderOutput(NamedTuple):
    sample: torch.Tensor
    latent: torch.Tensor
    posterior: Optional[DiagonalGaussianDistribution]


class CausalEncoderOutput(NamedTuple):
    latent: torch.Tensor
    posterior: Optional[DiagonalGaussianDistribution]


class CausalDecoderOutput(NamedTuple):
    sample: torch.Tensor
