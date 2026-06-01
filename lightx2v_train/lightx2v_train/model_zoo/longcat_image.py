from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, LongCatImagePipeline
from diffusers.models.transformers import LongCatImageTransformer2DModel
from diffusers.pipelines.longcat_image.pipeline_longcat_image import prepare_pos_ids

from lightx2v_train.utils.registry import MODEL_REGISTER

from .base import BaseModel


@dataclass
class LongCatImageDenoiserInput:
    hidden_states: torch.Tensor
    img_ids: torch.Tensor
    height: int
    width: int


@MODEL_REGISTER("longcat_image")
class LongCatImageModel(BaseModel):
    pipeline_cls = LongCatImagePipeline

    def load_components(self):
        model_path = self.config["model"]["pretrained_model_name_or_path"]
        self.text_pipeline = LongCatImagePipeline.from_pretrained(
            model_path,
            transformer=None,
            vae=None,
            torch_dtype=self.running_dtype,
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(self.device, dtype=self.running_dtype)
        self.transformer = LongCatImageTransformer2DModel.from_pretrained(model_path, subfolder="transformer").to(self.device, dtype=self.running_dtype)
        self.text_pipeline.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

    def denoiser_module(self):
        return self.transformer

    def fsdp2_shard_plan(self, fsdp_config):
        reshard_config = fsdp_config["reshard_after_forward"]
        return [
            {
                "modules": self.transformer.transformer_blocks,
                "reshard_after_forward": reshard_config["block_reshard"],
            },
            {
                "modules": self.transformer.single_transformer_blocks,
                "reshard_after_forward": reshard_config["block_reshard"],
            },
            {
                "module": self.transformer,
                "reshard_after_forward": reshard_config["root_reshard"],
            },
        ]

    @property
    def vae_scale_factor(self):
        return 2 ** (len(self.vae.config.block_out_channels) - 1)

    def encode_to_latent(self, sample):
        image = sample["target_image"].to(device=self.device, dtype=self.running_dtype)
        latent = self.vae.encode(image).latent_dist.sample()
        shift = getattr(self.vae.config, "shift_factor", 0.0)
        scale = getattr(self.vae.config, "scaling_factor", 1.0)
        return (latent - shift) * scale

    def encode_condition(self, sample):
        prompt = sample["prompt"]
        if self.config.get("enable_prompt_rewrite_training", False):
            prompt = self.text_pipeline.rewire_prompt(prompt, self.device)
        prompt_embed, text_ids = self.text_pipeline.encode_prompt(
            prompt=prompt,
            num_images_per_prompt=1,
        )
        return {"prompt_embed": prompt_embed, "text_ids": text_ids}

    def prepare_denoiser_input(self, noisy_latent):
        n = noisy_latent.shape[0]
        h, w = noisy_latent.shape[2], noisy_latent.shape[3]
        packed = LongCatImagePipeline._pack_latents(noisy_latent, n, noisy_latent.shape[1], h, w)
        img_ids = prepare_pos_ids(
            modality_id=1,
            type="image",
            start=(self.text_pipeline.tokenizer_max_length, self.text_pipeline.tokenizer_max_length),
            height=h // 2,
            width=w // 2,
        ).to(self.device)
        return LongCatImageDenoiserInput(
            hidden_states=packed,
            img_ids=img_ids,
            height=h,
            width=w,
        )

    def denoise(self, denoiser_input, timestep_or_sigma, condition):
        return self.transformer(
            hidden_states=denoiser_input.hidden_states,
            timestep=timestep_or_sigma,
            guidance=None,
            encoder_hidden_states=condition["prompt_embed"],
            txt_ids=condition["text_ids"],
            img_ids=denoiser_input.img_ids,
            return_dict=False,
        )[0]

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        return LongCatImagePipeline._unpack_latents(
            prediction,
            height=denoiser_input.height * self.vae_scale_factor,
            width=denoiser_input.width * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )

    def prepare_infer_latents(self, height, width, generator=None):
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        # latent shape: (batch=1, latent_channels, latent_h, latent_w)
        shape = (1, self.vae.config.latent_channels, latent_h, latent_w)
        return torch.randn(shape, generator=generator, device=self.device, dtype=self.running_dtype)

    def decode_latent(self, latent):
        # Reverse the normalization from encode_to_latent:
        # encode: normalized = (raw - shift) * scale
        # decode: raw = normalized / scale + shift
        shift = getattr(self.vae.config, "shift_factor", 0.0)
        scale = getattr(self.vae.config, "scaling_factor", 1.0)
        latent = latent / scale + shift

        image = self.vae.decode(latent).sample  # (B, C, H, W)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1).float().cpu().numpy()
        return [Image.fromarray((img * 255).round().astype(np.uint8)) for img in image]

    def assemble_pipeline(self, scheduler=None):
        return LongCatImagePipeline(
            tokenizer=self.text_pipeline.tokenizer,
            text_encoder=self.text_pipeline.text_encoder,
            text_processor=self.text_pipeline.text_processor,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=scheduler or self.text_pipeline.scheduler,
        ).to(self.device)

    def get_pipeline_infer_kwargs(self, infer_config):
        enable_cfg = infer_config.get("enable_cfg", False)
        return {
            "height": infer_config.get("height", infer_config.get("default_height", 1024)),
            "width": infer_config.get("width", infer_config.get("default_width", 1024)),
            "num_inference_steps": infer_config.get("num_inference_steps", 50),
            "guidance_scale": infer_config.get("cfg_guidance_scale", 4.0) if enable_cfg else 1.0,
            "enable_cfg_renorm": infer_config.get("enable_cfg_renorm", True),
            "cfg_renorm_min": infer_config.get("cfg_renorm_min", 0.0),
            "enable_prompt_rewrite": infer_config.get("enable_prompt_rewrite", True),
        }
