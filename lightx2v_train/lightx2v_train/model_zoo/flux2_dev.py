from dataclasses import dataclass

import torch
from diffusers import AutoencoderKLFlux2, Flux2Pipeline, Flux2Transformer2DModel
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

from lightx2v_train.utils.registry import MODEL_REGISTER

from .base import BaseModel


@dataclass
class Flux2DevDenoiserInput:
    hidden_states: torch.Tensor
    img_ids: torch.Tensor
    height: int
    width: int


@MODEL_REGISTER("flux2_dev")
class Flux2DevModel(BaseModel):
    pipeline_cls = Flux2Pipeline

    def load_components(self, transformer_only=False, reference_model=None):
        model_config = self.config["model"]
        model_path = model_config["pretrained_model_name_or_path"]
        self.guidance_scale = float(model_config.get("guidance_scale", 4.0))

        if transformer_only:
            if reference_model is not None:
                self.text_pipeline = reference_model.text_pipeline
                self.vae = reference_model.vae
                self.image_processor = reference_model.image_processor
                self.guidance_scale = reference_model.guidance_scale
            self.transformer = self.load_transformer()
            return

        self.text_pipeline = Flux2Pipeline.from_pretrained(
            model_path,
            transformer=None,
            vae=None,
            torch_dtype=self.running_dtype,
        ).to(self.device)
        self.vae = AutoencoderKLFlux2.from_pretrained(model_path, subfolder="vae", torch_dtype=self.running_dtype).to(self.device)
        self.transformer = self.load_transformer()

        self.text_pipeline.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

    def load_transformer(self):
        model_path = self.config["model"]["pretrained_model_name_or_path"]
        return Flux2Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=self.running_dtype).to(self.device)

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
        latents = Flux2Pipeline._patchify_latents(latents)
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(latents.device, latents.dtype)
        return (latents - latents_bn_mean) / latents_bn_std

    def _denormalize_patch_latents(self, latents):
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(latents.device, latents.dtype)
        latents = latents * latents_bn_std + latents_bn_mean
        return Flux2Pipeline._unpatchify_latents(latents)

    def encode_to_latent(self, sample):
        image = sample["inputs"]["target_image"].to(device=self.device, dtype=self.running_dtype)
        latent = self.vae.encode(image).latent_dist.sample()
        return self._normalize_patch_latents(latent)

    def encode_condition(self, sample):
        prompt = sample["conditioning"]["prompt"]
        return self.encode_prompt_condition(prompt)

    def encode_prompt_condition(self, prompt):
        model_config = self.config["model"]
        prompt_embed, text_ids = self.text_pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=model_config.get("max_sequence_length", 512),
            text_encoder_out_layers=tuple(model_config.get("text_encoder_out_layers", (10, 20, 30))),
        )
        return {"prompt_embed": prompt_embed, "text_ids": text_ids}

    def prepare_denoiser_input(self, noisy_latent, condition=None):
        h, w = noisy_latent.shape[2], noisy_latent.shape[3]
        packed = Flux2Pipeline._pack_latents(noisy_latent)
        img_ids = Flux2Pipeline._prepare_latent_ids(noisy_latent).to(self.device)
        return Flux2DevDenoiserInput(
            hidden_states=packed,
            img_ids=img_ids,
            height=h,
            width=w,
        )

    def denoise(self, denoiser_input, timestep_or_sigma, condition):
        guidance = torch.full(
            (denoiser_input.hidden_states.shape[0],),
            self.guidance_scale,
            device=self.device,
            dtype=torch.float32,
        )
        return self.transformer(
            hidden_states=denoiser_input.hidden_states,
            timestep=timestep_or_sigma,
            guidance=guidance,
            encoder_hidden_states=condition["prompt_embed"],
            txt_ids=condition["text_ids"],
            img_ids=denoiser_input.img_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        return Flux2Pipeline._unpack_latents_with_ids(prediction, denoiser_input.img_ids)

    def prepare_infer_latents(self, height, width, generator=None):
        latent_h = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_w = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (1, self.transformer.config.in_channels, latent_h // 2, latent_w // 2)
        return torch.randn(shape, generator=generator, device=self.device, dtype=self.running_dtype)

    def dmd_latent_shape(self, batch_size, height, width):
        latent_h = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_w = 2 * (int(width) // (self.vae_scale_factor * 2))
        return (int(batch_size), int(self.transformer.config.in_channels), latent_h // 2, latent_w // 2)

    def cfg_on_denoiser_output(self):
        return True

    def decode_latent(self, latent):
        latent = self._denormalize_patch_latents(latent)
        image = self.vae.decode(latent).sample
        return self.image_processor.postprocess(image, output_type="pil")

    def assemble_pipeline(self, scheduler=None):
        return Flux2Pipeline(
            tokenizer=self.text_pipeline.tokenizer,
            text_encoder=self.text_pipeline.text_encoder,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=scheduler or self.text_pipeline.scheduler,
        ).to(self.device)

    def get_pipeline_infer_kwargs(self, infer_config):
        return {
            "height": infer_config.get("height", infer_config.get("default_height", 1024)),
            "width": infer_config.get("width", infer_config.get("default_width", 1024)),
            "num_inference_steps": infer_config.get("num_inference_steps", 50),
            "guidance_scale": infer_config.get("cfg_guidance_scale", self.guidance_scale),
            "max_sequence_length": self.config["model"].get("max_sequence_length", 512),
            "text_encoder_out_layers": tuple(self.config["model"].get("text_encoder_out_layers", (10, 20, 30))),
        }
