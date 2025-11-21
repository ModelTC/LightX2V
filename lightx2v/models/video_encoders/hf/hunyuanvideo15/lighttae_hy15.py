import torch
import torch.nn as nn

from lightx2v.models.video_encoders.hf.tae import TAEHV


class LightTaeHy15(nn.Module):
    def __init__(self, vae_path="lighttae_hy1_5.pth", dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.taehv = TAEHV(vae_path, model_type="hy15", latent_channels=32, patch_size=2).to(self.dtype)
        self.scaling_factor = 1.03682

    @torch.no_grad()
    def decode(self, latents, parallel=True, show_progress_bar=True, skip_trim=False):
        latents = latents / self.scaling_factor
        return self.taehv.decode_video(latents.transpose(1, 2).to(self.dtype), parallel, show_progress_bar).transpose(1, 2)
