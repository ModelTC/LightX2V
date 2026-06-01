import math

import torch


def apply_timesteps_embedding(timesteps, num_channels, downscale_freq_shift=0.0, scale=1, max_period=10000):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    half_dim = num_channels // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if num_channels % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb
