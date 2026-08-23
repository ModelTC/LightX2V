from dataclasses import dataclass

import torch
from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

from ..base import BaseModel


@dataclass
class Flux2DenoiserInput:
    hidden_states: torch.Tensor
    image_ids: torch.Tensor
    target_ids: torch.Tensor
    target_token_length: int
    height: int
    width: int


class Flux2ModelBase(BaseModel):
    """Shared tensor mechanics for the Flux2 model family."""

    requires_source_images = False
    target_latent_mode = "sample"
    default_text_encoder_out_layers = ()

    def load_components(self, transformer_only=False, reference_model=None):
        model_path = self.config["model"]["pretrained_model_name_or_path"]
        self._validate_model_path(model_path)
        self._configure_model()

        if transformer_only:
            if reference_model is not None:
                self.text_pipeline = reference_model.text_pipeline
                self.vae = reference_model.vae
                self.image_processor = reference_model.image_processor
                self._copy_model_state(reference_model)
            self.transformer = self.load_transformer(model_path)
            return

        self.text_pipeline = self.pipeline_cls.from_pretrained(
            model_path,
            transformer=None,
            vae=None,
            torch_dtype=self.running_dtype,
        ).to(self.device)
        self.vae = AutoencoderKLFlux2.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=self.running_dtype,
        ).to(self.device)
        self.transformer = self.load_transformer(model_path)

        self.text_pipeline.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

    def _validate_model_path(self, model_path):
        del model_path

    def _configure_model(self):
        pass

    def _copy_model_state(self, reference_model):
        del reference_model

    def load_transformer(self, model_path=None):
        model_path = model_path or self.config["model"]["pretrained_model_name_or_path"]
        return Flux2Transformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=self.running_dtype,
        ).to(self.device)

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
        latents = self.pipeline_cls._patchify_latents(latents)
        mean, std = self._latent_statistics(latents)
        return (latents - mean) / std

    def _denormalize_patch_latents(self, latents):
        mean, std = self._latent_statistics(latents)
        return self.pipeline_cls._unpatchify_latents(latents * std + mean)

    def _latent_statistics(self, latents):
        mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        variance = self.vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        return mean, torch.sqrt(variance + self.vae.config.batch_norm_eps)

    def encode_to_latent(self, sample):
        image = sample["inputs"]["target_pixel_values"]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Expected target_pixel_values with shape [B, C, H, W], got {tuple(image.shape)}")
        image = image.to(device=self.device, dtype=self.running_dtype)
        distribution = self.vae.encode(image).latent_dist
        latent = getattr(distribution, self.target_latent_mode)()
        return self._normalize_patch_latents(latent)

    def encode_condition(self, sample):
        prompt = sample["conditioning"]["prompt"]
        if self.requires_source_images:
            return self.encode_conditions_with_source(sample, [prompt])[0]
        return self.encode_prompt_condition(prompt)

    def encode_prompt_condition(self, prompt):
        model_config = self.config["model"]
        prompt_embed, text_ids = self.text_pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=model_config.get("max_sequence_length", 512),
            text_encoder_out_layers=tuple(model_config.get("text_encoder_out_layers", self.default_text_encoder_out_layers)),
        )
        return {"prompt_embed": prompt_embed, "text_ids": text_ids}

    def encode_conditions_with_source(self, sample, prompts):
        images = sample["inputs"].get("source_vae_pixel_values", [])
        if not images:
            raise ValueError(f"{type(self).__name__} requires at least one source image")

        reference_tokens, reference_ids = self._encode_reference_images(images)
        conditions = []
        for prompt in prompts:
            condition = self.encode_prompt_condition(prompt)
            condition.update(reference_tokens=reference_tokens, reference_ids=reference_ids)
            conditions.append(condition)
        return conditions

    def _encode_reference_images(self, images):
        latents = []
        for image in images:
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.ndim != 4:
                raise ValueError(f"Expected source image with shape [B, C, H, W], got {tuple(image.shape)}")
            image = image.to(device=self.device, dtype=self.running_dtype)
            latent = self.vae.encode(image).latent_dist.mode()
            latents.append(self._normalize_patch_latents(latent))

        batch_size = latents[0].shape[0]
        if any(latent.shape[0] != batch_size for latent in latents):
            raise ValueError("All source images must have the same batch size")

        reference_ids = self.pipeline_cls._prepare_image_ids([latent[:1] for latent in latents])
        reference_ids = reference_ids.expand(batch_size, -1, -1).to(self.device)
        reference_tokens = torch.cat([self.pipeline_cls._pack_latents(latent) for latent in latents], dim=1)
        return reference_tokens, reference_ids

    def prepare_denoiser_input(self, noisy_latent, condition=None):
        height, width = noisy_latent.shape[-2:]
        target_tokens = self.pipeline_cls._pack_latents(noisy_latent)
        target_ids = self.pipeline_cls._prepare_latent_ids(noisy_latent).to(noisy_latent.device)

        hidden_states = target_tokens
        image_ids = target_ids
        if condition is not None and condition.get("reference_tokens") is not None:
            reference_tokens = condition["reference_tokens"]
            reference_ids = condition.get("reference_ids")
            if reference_ids is None:
                raise ValueError("Flux2 reference tokens require matching reference IDs")
            if reference_tokens.shape[0] != target_tokens.shape[0]:
                raise ValueError("Flux2 target and reference tokens must have the same batch size")
            reference_tokens = reference_tokens.to(target_tokens.device, target_tokens.dtype)
            reference_ids = reference_ids.to(target_ids.device)
            hidden_states = torch.cat([target_tokens, reference_tokens], dim=1)
            image_ids = torch.cat([target_ids, reference_ids], dim=1)

        return Flux2DenoiserInput(
            hidden_states=hidden_states,
            image_ids=image_ids,
            target_ids=target_ids,
            target_token_length=target_tokens.shape[1],
            height=height,
            width=width,
        )

    def _denoise(self, denoiser_input, timestep_or_sigma, condition, guidance):
        prediction = self.transformer(
            hidden_states=denoiser_input.hidden_states,
            timestep=timestep_or_sigma,
            guidance=guidance,
            encoder_hidden_states=condition["prompt_embed"],
            txt_ids=condition["text_ids"],
            img_ids=denoiser_input.image_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]
        return prediction[:, : denoiser_input.target_token_length]

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        return self.pipeline_cls._unpack_latents_with_ids(prediction, denoiser_input.target_ids)

    def prepare_infer_latents(self, height, width, generator=None):
        latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (1, self.transformer.config.in_channels, latent_height // 2, latent_width // 2)
        return torch.randn(shape, generator=generator, device=self.device, dtype=self.running_dtype)

    def decode_latent(self, latent):
        image = self.vae.decode(self._denormalize_patch_latents(latent)).sample
        return self.image_processor.postprocess(image, output_type="pil")
