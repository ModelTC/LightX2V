import gc
import json
import os
from typing import Optional

import torch

try:
    from diffusers import AutoencoderKLQwenImage
    from diffusers.image_processor import VaeImageProcessor
except ImportError:
    AutoencoderKLQwenImage = None
    VaeImageProcessor = None


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class AutoencoderKLQwenImageVAE:
    def __init__(self, config):
        self.config = config

        self.cpu_offload = config.get("cpu_offload", False)
        if self.cpu_offload:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.latent_channels = config["vae_z_dim"]
        self.load()

    def load(self):
        self.model = AutoencoderKLQwenImage.from_pretrained(os.path.join(self.config["model_path"], "vae")).to(self.device).to(self.dtype)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.config["vae_scale_factor"] * 2)
        with open(os.path.join(self.config["model_path"], "vae", "config.json"), "r") as f:
            vae_config = json.load(f)
            self.vae_scale_factor = 2 ** len(vae_config["temperal_downsample"]) if "temperal_downsample" in vae_config else 8

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batchsize, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batchsize, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batchsize, channels // (2 * 2), 1, height, width)

        return latents

    @torch.no_grad()
    def decode(self, latents, input_info):
        if self.cpu_offload:
            self.model.to(torch.device("cuda"))
        if self.config["task"] == "t2i":
            width, height = self.config["aspect_ratios"][self.config["aspect_ratio"]]
        elif self.config["task"] == "i2i":
            width, height = input_info.auto_width, input_info.auto_hight
        latents = self._unpack_latents(latents, height, width, self.config["vae_scale_factor"])
        latents = latents.to(self.dtype)
        latents_mean = torch.tensor(self.config["vae_latents_mean"]).view(1, self.config["vae_z_dim"], 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(self.config["vae_latents_std"]).view(1, self.config["vae_z_dim"], 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        images = self.model.decode(latents, return_dict=False)[0][:, :, 0]
        images = self.image_processor.postprocess(images, output_type="pil")
        if self.cpu_offload:
            self.model.to(torch.device("cpu"))
            torch.cuda.empty_cache()
            gc.collect()
        return images

    @staticmethod
    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._pack_latents
    def _pack_latents(latents, batchsize, num_channels_latents, height, width):
        latents = latents.view(batchsize, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batchsize, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [retrieve_latents(self.model.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax") for i in range(image.shape[0])]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.model.encode(image), generator=generator, sample_mode="argmax")
        latents_mean = torch.tensor(self.model.config["latents_mean"]).view(1, self.latent_channels, 1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_std = torch.tensor(self.model.config["latents_std"]).view(1, self.latent_channels, 1, 1, 1).to(image_latents.device, image_latents.dtype)
        image_latents = (image_latents - latents_mean) / latents_std

        return image_latents

    @torch.no_grad()
    def encode_vae_image(self, image, input_info):
        self.generator = torch.Generator().manual_seed(input_info.seed)
        if self.cpu_offload:
            self.model.to(torch.device("cuda"))
        num_channels_latents = self.config["transformer_in_channels"] // 4
        image = image.to(self.model.device).to(self.dtype)

        if image.shape[1] != self.latent_channels:
            image_latents = self._encode_vae_image(image=image, generator=self.generator)
        else:
            image_latents = image
        if self.config["batchsize"] > image_latents.shape[0] and self.config["batchsize"] % image_latents.shape[0] == 0:
            # expand init_latents for batchsize
            additional_image_per_prompt = self.config["batchsize"] // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif self.config["batchsize"] > image_latents.shape[0] and self.config["batchsize"] % image_latents.shape[0] != 0:
            raise ValueError(f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {self.config['batchsize']} text prompts.")
        else:
            image_latents = torch.cat([image_latents], dim=0)

        image_latent_height, image_latent_width = image_latents.shape[3:]
        image_latents = self._pack_latents(image_latents, self.config["batchsize"], num_channels_latents, image_latent_height, image_latent_width)
        if self.cpu_offload:
            self.model.to(torch.device("cpu"))
            torch.cuda.empty_cache()
            gc.collect()
        return image_latents
