import os

import torch
from PIL import Image

from .autoencoder import load_ae


class BagelVae:
    def __init__(self, config):
        self.config = config
        vae_path = os.path.join(config["model_path"], "ae.safetensors")
        self.vae_model, self.vae_params = load_ae(vae_path)
        self.vae_model = self.vae_model

    def decode(self, latents, decode_info):
        latents = latents.split((decode_info["packed_seqlens"] - 2).tolist())
        data = torch.load("/data/nvme2/wushuo/Bagel/unpacked_latent.pt")

        H, W = decode_info["image_shape"]
        h, w = H // decode_info["latent_downsample"], W // decode_info["latent_downsample"]

        latents = latents[0]
        latents = latents.reshape(1, h, w, decode_info["latent_patch_size"], decode_info["latent_patch_size"], decode_info["latent_channel"])
        latents = torch.einsum("nhwpqc->nchpwq", latents)
        latents = latents.reshape(1, decode_info["latent_channel"], h * decode_info["latent_patch_size"], w * decode_info["latent_patch_size"])

        image = self.vae_model.decode(latents)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())
        return [image]
