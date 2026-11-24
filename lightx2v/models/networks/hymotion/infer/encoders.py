# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from __future__ import annotations

from lightx2v.models.networks.hymotion.weights.encoders import Tensor, modulate, torch


def MLPEncoder_forward(self, x: Tensor) -> Tensor:
    return self.linears(x)


def FinalLayer_forward(self, x: Tensor, adapter: Tensor) -> Tensor:
    shift, scale = self.adaLN_modulation(adapter).chunk(2, dim=-1)
    x = modulate(self.norm_final(x), shift=shift, scale=scale)
    x = self.linear(x)
    return x


def TimestepEmbeddingEncoder_forward(self, t: Tensor) -> Tensor:
    x = self.blocks(self.sinusodial_embedding(t, self.embedding_dim, time_factor=self.time_factor)).unsqueeze(1)
    return x


def TimestepEmbeddingEncoder_sinusodial_embedding(timesteps: Tensor, embedding_dim: int, temperature: float = 10000.0, time_factor: float = 1.0) -> Tensor:
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    timesteps = timesteps * time_factor
    half = embedding_dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(temperature)) * torch.arange(start=0, end=half, dtype=torch.float) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embedding_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
