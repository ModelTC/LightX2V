from dataclasses import dataclass

import torch
from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel

from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from .base import BaseModel


@dataclass
class WanT2VDenoiserInput:
    hidden_states: torch.Tensor


@MODEL_REGISTER("wan_t2v")
class WanT2VModel(BaseModel):
    pipeline_cls = WanPipeline

    def load_components(self):
        model_config = self.config["model"]
        model_path = model_config["pretrained_model_name_or_path"]

        self.load_vae = model_config.get("load_vae", True)
        self.load_text_encoder = model_config.get("load_text_encoder", True)
        self.load_transformer = model_config.get("load_transformer", True)
        self.sample_posterior = model_config.get("sample_posterior", True)
        self.num_train_timesteps = self.config.get("scheduler", {}).get("num_train_timesteps", 1000)
        self.max_sequence_length = model_config.get("max_sequence_length", 512)
        self.vae_dtype = get_running_dtype(model_config.get("vae_dtype", "fp32"))

        if self.load_vae:
            self.vae = AutoencoderKLWan.from_pretrained(
                model_path,
                subfolder="vae",
                torch_dtype=self.vae_dtype,
            ).to(self.device)
            self.vae.requires_grad_(False)
            if model_config.get("enable_vae_tiling", False):
                self.vae.enable_tiling()

        if self.load_text_encoder:
            if self.vae is None:
                raise ValueError("WanT2VModel requires load_vae=True when load_text_encoder=True.")
            self.text_pipeline = WanPipeline.from_pretrained(
                model_path,
                vae=self.vae,
                transformer=None,
                torch_dtype=self.running_dtype,
            ).to(self.device)
            self.text_pipeline.text_encoder.requires_grad_(False)
        else:
            self.text_pipeline = None

        if self.load_transformer:
            self.transformer = WanTransformer3DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=self.running_dtype,
            ).to(self.device)
        else:
            self.transformer = None

        if self.vae is not None:
            self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal
            self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial
        else:
            self.vae_scale_factor_temporal = model_config.get("vae_scale_factor_temporal", 4)
            self.vae_scale_factor_spatial = model_config.get("vae_scale_factor_spatial", 8)

    def denoiser_module(self):
        if self.transformer is None:
            raise RuntimeError("Wan transformer is not loaded. Set model.load_transformer=True for training.")
        return self.transformer

    def fsdp2_shard_plan(self, fsdp_config):
        reshard_config = fsdp_config.get(
            "reshard_after_forward",
            {
                "root_reshard": False,
                "block_reshard": True,
            },
        )
        return [
            {
                "modules": self.transformer.blocks,
                "reshard_after_forward": reshard_config.get("block_reshard", True),
            },
            {
                "module": self.transformer,
                "reshard_after_forward": reshard_config.get("root_reshard", False),
            },
        ]

    def encode_to_latent(self, sample):
        if "latent" in sample:
            latent = sample["latent"].to(device=self.device, dtype=self.running_dtype)
            if latent.ndim == 4:
                latent = latent.unsqueeze(0)
            return latent

        if self.vae is None:
            raise RuntimeError("Wan VAE is not loaded. Use cached latents or set model.load_vae=True.")

        video = sample["video"].to(device=self.device, dtype=self.vae_dtype)
        posterior = self.vae.encode(video).latent_dist
        latent = posterior.sample() if self.sample_posterior else posterior.mode()

        latent_mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=self.vae_dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latent_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=self.device, dtype=self.vae_dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latent = (latent - latent_mean) * latent_std
        return latent.to(dtype=self.running_dtype)

    def encode_condition(self, sample):
        if "prompt_embed" in sample:
            prompt_embed = sample["prompt_embed"].to(device=self.device, dtype=self.running_dtype)
            if prompt_embed.ndim == 2:
                prompt_embed = prompt_embed.unsqueeze(0)
            return {"prompt_embed": prompt_embed}

        if self.text_pipeline is None:
            raise RuntimeError("Wan text encoder is not loaded. Use cached prompt embeds or set model.load_text_encoder=True.")

        prompt_embed, _ = self.text_pipeline.encode_prompt(
            prompt=sample["prompt"],
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=self.max_sequence_length,
            device=self.device,
            dtype=self.running_dtype,
        )
        return {"prompt_embed": prompt_embed.to(dtype=self.running_dtype)}

    def prepare_denoiser_input(self, noisy_latent):
        return WanT2VDenoiserInput(hidden_states=noisy_latent)

    def denoise(self, denoiser_input, timestep_or_sigma, condition):
        timestep = timestep_or_sigma.float() * self.num_train_timesteps
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.to(device=self.device)

        return self.transformer(
            hidden_states=denoiser_input.hidden_states,
            timestep=timestep,
            encoder_hidden_states=condition["prompt_embed"],
            return_dict=False,
        )[0]

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        return prediction

    def prepare_infer_latents(self, height, width, generator=None):
        num_frames = self.config.get("inference", {}).get("num_frames", 81)
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            1,
            self.transformer.config.in_channels,
            num_latent_frames,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        return torch.randn(shape, generator=generator, device=self.device, dtype=self.running_dtype)

    def decode_latent(self, latent):
        if self.vae is None:
            raise RuntimeError("Wan VAE is not loaded. Set model.load_vae=True for decoding.")

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=self.vae_dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=self.device, dtype=self.vae_dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        latent = latent.to(dtype=self.vae_dtype) / latents_std + latents_mean
        video = self.vae.decode(latent, return_dict=False)[0]
        if self.text_pipeline is not None:
            return self.text_pipeline.video_processor.postprocess_video(video, output_type="pil")
        return video

    def assemble_pipeline(self, scheduler=None):
        if self.vae is None or self.text_pipeline is None:
            raise RuntimeError("Wan VAE and text encoder must be loaded to assemble an inference pipeline.")

        return WanPipeline(
            tokenizer=self.text_pipeline.tokenizer,
            text_encoder=self.text_pipeline.text_encoder,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=scheduler or self.text_pipeline.scheduler,
        ).to(self.device)

    def get_pipeline_infer_kwargs(self, infer_config):
        return {
            "height": infer_config.get("height", infer_config.get("default_height", 480)),
            "width": infer_config.get("width", infer_config.get("default_width", 832)),
            "num_frames": infer_config.get("num_frames", 81),
            "num_inference_steps": infer_config.get("num_inference_steps", 50),
            "guidance_scale": infer_config.get("cfg_guidance_scale", 5.0),
        }
