"""Krea's tensor-cache policy for the existing Wan2.1 temporal upsamplers."""

import torch
from einops import rearrange

from lightx2v.models.video_encoders.hf.wan.vae import Resample


class KreaTemporalResample(Resample):
    """Match Krea demo_utils.vae_block3 without duplicating the Wan codec.

    Its two-frame history starts at zero. For a one-frame update it appends
    zero where the prior tail was zero, otherwise repeating the current frame.
    Standard Wan instead retains the preceding frame; that difference affects
    later RGB chunks and therefore Krea's pixel-to-context feedback.
    """

    def forward(self, x, feat_cache=None, feat_idx=None):
        if self.mode != "upsample3d" or feat_cache is None:
            return super().forward(x, feat_cache, [0] if feat_idx is None else feat_idx)
        if feat_idx is None:
            feat_idx = [0]
        batch, channels, frames, height, width = x.shape
        index = feat_idx[0]
        previous = feat_cache[index]
        if previous is None:
            feat_cache[index] = x.new_zeros(batch, channels, 2, height, width)
        else:
            retained = x[:, :, -2:].clone()
            if retained.shape[2] < 2:
                padding = torch.where(previous[:, :, -1:] == 0, 0, retained)
                retained = torch.cat((padding, retained), dim=2)
            x = self.time_conv(x, previous)
            feat_cache[index] = retained
            x = x.reshape(batch, 2, channels, frames, height, width)
            x = torch.stack((x[:, 0], x[:, 1]), dim=3).reshape(batch, channels, frames * 2, height, width)
        feat_idx[0] += 1
        frames = x.shape[2]
        x = self.resample(rearrange(x, "b c t h w -> (b t) c h w"))
        return rearrange(x, "(b t) c h w -> b c t h w", t=frames)


def configure_realtime_video_vae(vae):
    """Keep the loaded Wan state and replace only temporal decoder policies."""
    for module in vae.model.decoder.modules():
        if type(module) is Resample and module.mode == "upsample3d":
            module.__class__ = KreaTemporalResample
    return vae
