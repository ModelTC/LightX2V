import torch
import torch.nn as nn

from lightx2v.models.video_encoders.hf.tae import TAEHV
from lightx2v.utils.memory_profiler import peak_memory_decorator


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class WanVAE_tiny(nn.Module):
    def __init__(self, vae_path="taew2_1.pth", dtype=torch.bfloat16, device="cuda", need_scaled=False):
        super().__init__()
        self.dtype = dtype
        self.device = torch.device("cuda")
        self.taehv = TAEHV(vae_path).to(self.dtype)
        self.temperal_downsample = [True, True, False]
        self.need_scaled = need_scaled

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

    @torch.no_grad()
    def encode_video(self, vid):
        return self.taehv.encode_video(vid, parallel=False)

    @torch.no_grad()
    def decode_video(self, vid_enc):
        return self.taehv.decode_video(vid_enc, parallel=False)


class Wan2_2_VAE_tiny(nn.Module):
    def __init__(self, vae_path="taew2_2.pth", dtype=torch.bfloat16, device="cuda", need_scaled=False):
        super().__init__()
        self.dtype = dtype
        self.device = torch.device("cuda")
        self.taehv = TAEHV(vae_path, model_type="wan22").to(self.dtype)
        self.need_scaled = need_scaled
        if self.need_scaled:
            self.latents_mean = [
                -0.2289,
                -0.0052,
                -0.1323,
                -0.2339,
                -0.2799,
                0.0174,
                0.1838,
                0.1557,
                -0.1382,
                0.0542,
                0.2813,
                0.0891,
                0.1570,
                -0.0098,
                0.0375,
                -0.1825,
                -0.2246,
                -0.1207,
                -0.0698,
                0.5109,
                0.2665,
                -0.2108,
                -0.2158,
                0.2502,
                -0.2055,
                -0.0322,
                0.1109,
                0.1567,
                -0.0729,
                0.0899,
                -0.2799,
                -0.1230,
                -0.0313,
                -0.1649,
                0.0117,
                0.0723,
                -0.2839,
                -0.2083,
                -0.0520,
                0.3748,
                0.0152,
                0.1957,
                0.1433,
                -0.2944,
                0.3573,
                -0.0548,
                -0.1681,
                -0.0667,
            ]

            self.latents_std = [
                0.4765,
                1.0364,
                0.4514,
                1.1677,
                0.5313,
                0.4990,
                0.4818,
                0.5013,
                0.8158,
                1.0344,
                0.5894,
                1.0901,
                0.6885,
                0.6165,
                0.8454,
                0.4978,
                0.5759,
                0.3523,
                0.7135,
                0.6804,
                0.5833,
                1.4146,
                0.8986,
                0.5659,
                0.7069,
                0.5338,
                0.4889,
                0.4917,
                0.4069,
                0.4999,
                0.6866,
                0.4093,
                0.5709,
                0.6065,
                0.6415,
                0.4944,
                0.5726,
                1.2042,
                0.5458,
                1.6887,
                0.3971,
                1.0600,
                0.3943,
                0.5537,
                0.5444,
                0.4089,
                0.7468,
                0.7744,
            ]

            self.z_dim = 48

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

    @torch.no_grad()
    def encode_video(self, vid):
        return self.taehv.encode_video(vid, parallel=False)

    @torch.no_grad()
    def decode_video(self, vid_enc):
        return self.taehv.decode_video(vid_enc, parallel=False)
