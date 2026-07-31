import os

import torch
from PIL import Image

from lightx2v_platform.base.global_var import AI_DEVICE

from .autoencoder import load_ae


class BagelVae:
    def __init__(self, config):
        self.config = config
        vae_path = os.path.join(config["model_path"], "ae.safetensors")
        if not os.path.exists(vae_path):
            raise FileNotFoundError(f"BAGEL VAE weights not found: {vae_path}. Expected `ae.safetensors` in model_path.")
        self.vae_model, self.vae_params = load_ae(vae_path)
        self.device = torch.device(AI_DEVICE)
        self.vae_model = self.vae_model.to(device=self.device, dtype=torch.float32).eval()

    def encode(self, images):
        images = images.to(device=self.device, dtype=torch.float32)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            return self.vae_model.encode(images)

    def decode(self, latents, decode_info):
        latents = latents.split((decode_info["packed_seqlens"] - 2).tolist())

        H, W = decode_info["image_shape"]
        h, w = H // decode_info["latent_downsample"], W // decode_info["latent_downsample"]

        latents = latents[0]
        latents = latents.reshape(1, h, w, decode_info["latent_patch_size"], decode_info["latent_patch_size"], decode_info["latent_channel"])
        latents = torch.einsum("nhwpqc->nchpwq", latents)
        latents = latents.reshape(1, decode_info["latent_channel"], h * decode_info["latent_patch_size"], w * decode_info["latent_patch_size"])

        latents = latents.to(device=self.device, dtype=torch.float32)
        image = self.vae_model.decode(latents)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = image.to(torch.uint8).cpu()
        if decode_info.get("return_result_tensor", False):
            return [image]
        return [Image.fromarray(image.numpy())]
