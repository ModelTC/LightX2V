import math

import torch


def get_timestep_embedding(
    timesteps,
    embedding_dim=256,
    flip_sin_to_cos=True,
    downscale_freq_shift=0,
    scale=1,
    max_period=10000,
):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0,
        end=half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb(x, freqs_cis):
    with torch.amp.autocast(x.device.type, enabled=False):
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        out = torch.view_as_real(x_c * freqs_cis.unsqueeze(1)).flatten(2)
        return out.type_as(x)


def precompute_freqs_cis(dim, end, theta):
    freqs_cis = []
    for d, e in zip(dim, end):
        freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
        timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
        freqs = torch.outer(timestep, freqs).float()
        freqs_cis.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
    return freqs_cis


class LingBotVideoRotaryEmbedding:
    def __init__(self, axes_dims, axes_lens, theta):
        self.axes_dims = tuple(axes_dims)
        self.axes_lens = list(axes_lens)
        self.theta = theta
        self.freqs_cis = None

    def __call__(self, position_ids):
        device = position_ids.device
        max_vals = position_ids.max(dim=0).values.tolist()
        needs_rebuild = self.freqs_cis is None or any(max_val >= axis_len for max_val, axis_len in zip(max_vals, self.axes_lens))
        if needs_rebuild:
            for i in range(len(self.axes_lens)):
                if max_vals[i] >= self.axes_lens[i]:
                    self.axes_lens[i] = int(max_vals[i] * 1.5) + 1
            self.freqs_cis = precompute_freqs_cis(self.axes_dims, tuple(self.axes_lens), self.theta)
            self.freqs_cis = [freqs.to(device) for freqs in self.freqs_cis]
        elif self.freqs_cis[0].device != device:
            self.freqs_cis = [freqs.to(device) for freqs in self.freqs_cis]
        return torch.cat([self.freqs_cis[i][position_ids[:, i]] for i in range(len(self.axes_dims))], dim=-1)


def make_joint_position_ids(text_len, grid_t, grid_h, grid_w, device):
    tt = torch.arange(grid_t, device=device, dtype=torch.int32) + (text_len + 1)
    hh = torch.arange(grid_h, device=device, dtype=torch.int32)
    ww = torch.arange(grid_w, device=device, dtype=torch.int32)
    grid = torch.stack(torch.meshgrid(tt, hh, ww, indexing="ij"), dim=-1).flatten(0, 2)
    text_t = torch.arange(text_len, device=device, dtype=torch.int32) + 1
    text_pos = torch.stack([text_t, torch.zeros_like(text_t), torch.zeros_like(text_t)], dim=-1)
    return torch.cat([grid, text_pos], dim=0)
