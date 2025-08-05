import torch
import torch.nn as nn

from lightx2v.utils.memory_profiler import peak_memory_decorator

from ..tae import TAEHV


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class WanVAE_tiny(nn.Module):
    def __init__(self, vae_pth="taew2_1.pth", dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.dtype = dtype
        self.device = torch.device("cuda")
        self.taehv = TAEHV(vae_pth).to(self.dtype)
        self.temperal_downsample = [True, True, False]
        self.config = DotDict(scaling_factor=1.0, latents_mean=torch.zeros(16), z_dim=16, latents_std=torch.ones(16))

    @peak_memory_decorator
    @torch.no_grad()
    def decode(self, latents, generator=None, return_dict=None, config=None):
        latents = latents.unsqueeze(0)
        n, c, t, h, w = latents.shape
        # low-memory, set parallel=True for faster + higher memory
        return self.taehv.decode_video(latents.transpose(1, 2).to(self.dtype), parallel=False).transpose(1, 2).mul_(2).sub_(1)
