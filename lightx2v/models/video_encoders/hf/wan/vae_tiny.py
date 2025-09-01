import torch
import torch.nn as nn

from lightx2v.utils.memory_profiler import peak_memory_decorator

from ..tae import TAEHV


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class WanVAE_tiny(nn.Module):
    def __init__(self, vae_pth="taew2_1.pth", dtype=torch.bfloat16, device="cuda", need_scaled=False):
        super().__init__()
        self.dtype = dtype
        self.device = torch.device("cuda")
        self.taehv = TAEHV(vae_pth).to(self.dtype)
        self.temperal_downsample = [True, True, False]
        self.config = DotDict(scaling_factor=1.0, latents_mean=torch.zeros(16), z_dim=16, latents_std=torch.ones(16))
        self.need_scaled = need_scaled

        # temp
        if self.need_scaled:
            self.latents_mean = [
                -0.7571,
                -0.7089,
                -0.9113,
                0.1075,
                -0.1745,
                0.9653,
                -0.1517,
                1.5508,
                0.4134,
                -0.0715,
                0.5517,
                -0.3632,
                -0.1922,
                -0.9497,
                0.2503,
                -0.2921,
            ]

            self.latents_std = [
                2.8184,
                1.4541,
                2.3275,
                2.6558,
                1.2196,
                1.7708,
                2.6052,
                2.0743,
                3.2687,
                2.1526,
                2.8652,
                1.5579,
                1.6382,
                1.1253,
                2.8251,
                1.9160,
            ]

            self.z_dim = 16

    @peak_memory_decorator
    @torch.no_grad()
    def decode(self, latents):
        latents = latents.unsqueeze(0)

        if self.need_scaled:
            latents_mean = torch.tensor(self.latents_mean).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents_std = 1.0 / torch.tensor(self.latents_std).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean

        # low-memory, set parallel=True for faster + higher memory
        return self.taehv.decode_video(latents.transpose(1, 2).to(self.dtype), parallel=False).transpose(1, 2).mul_(2).sub_(1)
