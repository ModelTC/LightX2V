from dataclasses import dataclass

import torch
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

from lightx2v_train.utils.registry import MODEL_REGISTER

from .base import BaseModel


@dataclass
class Flux2KleinDenoiserInput:
    hidden_states: torch.Tensor
    img_ids: torch.Tensor
    height: int
    width: int


@MODEL_REGISTER("flux2_klein")
class Flux2KleinModel(BaseModel):
    pipeline_cls = Flux2KleinPipeline

    def load_components(self):
        model_path = self.config["model"]["pretrained_model_name_or_path"]
        self.text_pipeline = Flux2KleinPipeline.from_pretrained(
            model_path,
            transformer=None,
            vae=None,
            torch_dtype=self.running_dtype,
        ).to(self.device)
        self.vae = AutoencoderKLFlux2.from_pretrained(model_path, subfolder="vae").to(self.device, dtype=self.running_dtype)
        self.transformer = Flux2Transformer2DModel.from_pretrained(model_path, subfolder="transformer").to(self.device, dtype=self.running_dtype)

        self.text_pipeline.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

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

    def _normalize_patch_latents(self, latents):
        latents = Flux2KleinPipeline._patchify_latents(latents)
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(latents.device, latents.dtype)
        return (latents - latents_bn_mean) / latents_bn_std

    def _denormalize_patch_latents(self, latents):
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(latents.device, latents.dtype)
        latents = latents * latents_bn_std + latents_bn_mean
        return Flux2KleinPipeline._unpatchify_latents(latents)

    def encode_to_latent(self, sample):
        image = sample["target_image"].to(device=self.device, dtype=self.running_dtype)
        latent = self.vae.encode(image).latent_dist.sample()
        return self._normalize_patch_latents(latent)

    def encode_condition(self, sample):
        prompt = sample["prompt"]
        model_config = self.config["model"]
        prompt_embed, text_ids = self.text_pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=model_config.get("max_sequence_length", 512),
            text_encoder_out_layers=tuple(model_config.get("text_encoder_out_layers", (9, 18, 27))),
        )
        return {"prompt_embed": prompt_embed, "text_ids": text_ids}

    def prepare_denoiser_input(self, noisy_latent):
        h, w = noisy_latent.shape[2], noisy_latent.shape[3]
        packed = Flux2KleinPipeline._pack_latents(noisy_latent)
        img_ids = Flux2KleinPipeline._prepare_latent_ids(noisy_latent).to(self.device)
        return Flux2KleinDenoiserInput(
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
            joint_attention_kwargs={},
            return_dict=False,
        )[0]

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        return Flux2KleinPipeline._unpack_latents_with_ids(
            prediction,
            denoiser_input.img_ids,
            height=denoiser_input.height,
            width=denoiser_input.width,
        )

    def prepare_infer_latents(self, height, width, generator=None):
        latent_h = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_w = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (1, self.transformer.config.in_channels, latent_h // 2, latent_w // 2)
        return torch.randn(shape, generator=generator, device=self.device, dtype=self.running_dtype)

    def decode_latent(self, latent):
        latent = self._denormalize_patch_latents(latent)
        image = self.vae.decode(latent).sample
        return self.image_processor.postprocess(image, output_type="pil")

    def assemble_pipeline(self, scheduler=None):
        return Flux2KleinPipeline(
            tokenizer=self.text_pipeline.tokenizer,
            text_encoder=self.text_pipeline.text_encoder,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=scheduler or self.text_pipeline.scheduler,
            is_distilled=self.text_pipeline.config.is_distilled,
        ).to(self.device)

    def get_pipeline_infer_kwargs(self, infer_config):
        enable_cfg = infer_config.get("enable_cfg", True)
        return {
            "height": infer_config.get("height", infer_config.get("default_height", 1024)),
            "width": infer_config.get("width", infer_config.get("default_width", 1024)),
            "num_inference_steps": infer_config.get("num_inference_steps", 50),
            "guidance_scale": infer_config.get("cfg_guidance_scale", 4.0) if enable_cfg else 1.0,
            "max_sequence_length": self.config["model"].get("max_sequence_length", 512),
            "text_encoder_out_layers": tuple(self.config["model"].get("text_encoder_out_layers", (9, 18, 27))),
        }
