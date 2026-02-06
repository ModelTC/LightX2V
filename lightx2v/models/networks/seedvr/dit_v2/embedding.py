from typing import Optional, Union

import torch
from diffusers.models.embeddings import get_timestep_embedding
from torch import nn


def emb_add(emb1: torch.Tensor, emb2: Optional[torch.Tensor]):
    return emb1 if emb2 is None else emb1 + emb2


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        sinusoidal_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.proj_in = nn.Linear(sinusoidal_dim, hidden_dim)
        self.proj_hid = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.SiLU()

    def forward(
        self,
        timestep: Union[int, float, torch.IntTensor, torch.FloatTensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.FloatTensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=dtype)
        if timestep.ndim == 0:
            timestep = timestep[None]

        emb = get_timestep_embedding(
            timesteps=timestep,
            embedding_dim=self.sinusoidal_dim,
            flip_sin_to_cos=False,
            downscale_freq_shift=0,
        )
        emb = emb.to(dtype)
        emb = self.proj_in(emb)
        emb = self.act(emb)
        emb = self.proj_hid(emb)
        emb = self.act(emb)
        emb = self.proj_out(emb)
        return emb
