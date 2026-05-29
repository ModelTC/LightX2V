import gc

import torch
from PIL import Image


def _truthy(value):
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


class AutoencoderKLErnieImageVAE:
    def __init__(self, config, model, unpatchify_latents=None, diffusers_cpu_offload=False):
        self.config = config
        self.model = model
        self.unpatchify_latents = unpatchify_latents or self._unpatchify_latents
        self.cpu_offload = _truthy(config.get("vae_cpu_offload", config.get("cpu_offload", False)))
        self.diffusers_cpu_offload = _truthy(diffusers_cpu_offload)
        self.manage_cpu_offload = self.cpu_offload and not self.diffusers_cpu_offload
        if self.manage_cpu_offload:
            self._move_model_to(torch.device("cpu"))

    def _move_model_to(self, device):
        if hasattr(self.model, "to"):
            self.model.to(device)

    def _empty_cache(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _unpatchify_latents(latents):
        batch_size, channels, height, width = latents.shape
        latents = latents.reshape(batch_size, channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(batch_size, channels // 4, height * 2,  width * 2)

    @torch.no_grad()
    def decode(self, latents, output_type="pil"):
        if output_type == "latent":
            return latents

        try:
            if self.manage_cpu_offload:
                self._move_model_to(latents.device)

            device = latents.device
            bn_mean = self.model.bn.running_mean.view(1, -1, 1, 1).to(device)
            bn_std = torch.sqrt(self.model.bn.running_var.view(1, -1, 1, 1).to(device) + 1e-5)
            latents = latents * bn_std + bn_mean
            latents = self.unpatchify_latents(latents)

            images = self.model.decode(latents, return_dict=False)[0]
            images = (images.clamp(-1, 1) + 1) / 2

            if output_type == "pt":
                return images
            if output_type == "pil":
                images = images.cpu().permute(0, 2, 3, 1).float().numpy()
                return [Image.fromarray((image * 255).astype("uint8")) for image in images]
            raise ValueError(f"Unsupported ERNIE-Image output_type '{output_type}'. Supported values: pil, pt, latent.")
        finally:
            if self.manage_cpu_offload:
                self._move_model_to(torch.device("cpu"))
                self._empty_cache()
