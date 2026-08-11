"""DMD distillation for MiniMax-H3 text-to-audio-video generation."""

import torch
from loguru import logger

from lightx2v_train.model_zoo.native.minimax_h3 import (
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    video_latent_num_frames,
)
from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .math import dmd_loss_pair, weighted_mse_pair
from .video_trainer import VideoDmdTrainer


def _shift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


@TRAINER_REGISTER("minimax_h3_t2av_dmd")
class MiniMaxH3T2AVDmdTrainer(VideoDmdTrainer):
    """Joint video/audio DMD with H3's data-ward velocity convention."""

    trainer_name = "minimax_h3_t2av_dmd"
    allowed_model_names = {"minimax_h3_t2av"}
    default_negative_prompt = ""
    default_lora_target_modules = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")
    supports_diversity_loss = False
    supports_real_data_fake = False

    def __init__(self, config):
        super().__init__(config)
        if self.cdm_enabled:
            raise ValueError("minimax_h3_t2av_dmd does not support CDM.")
        if self.guidance_scale > 1.0:
            raise ValueError("MiniMax-H3 is guidance-distilled and has no unconditional branch; set training.teacher.guidance_scale to 1.0.")
        h3 = self.training_config.get("minimax_h3", {})
        self.video_loss_weight = float(h3.get("video_loss_weight", 1.0))
        self.audio_loss_weight = float(h3.get("audio_loss_weight", 1.0))
        self.video_shift = float(h3.get("video_flow_shift", 12.0))
        self.audio_shift = float(h3.get("audio_flow_shift", 3.0))
        if self.video_shift <= 0 or self.audio_shift <= 0:
            raise ValueError("MiniMax-H3 flow shifts must be positive.")
        dtype_name = str(h3.get("latent_dtype", "fp32")).lower()
        latent_dtypes = {"fp32": torch.float32, "float32": torch.float32, "bf16": torch.bfloat16}
        if dtype_name not in latent_dtypes:
            raise ValueError(f"Unsupported training.minimax_h3.latent_dtype={dtype_name!r}.")
        self.latent_dtype = latent_dtypes[dtype_name]
        self._layout_cache = {}

        # DMD's num_inference_steps means model evaluations. H3's published
        # scheduler has one additional terminal-zero sigma grid point.
        base = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, dtype=torch.float32)
        self.video_sigmas_cpu = _shift_sigma(base, self.video_shift)
        self.audio_sigmas_cpu = _shift_sigma(base, self.audio_shift)
        logger.info(
            "[train] H3 DMD rollout evaluations={} video_shift={} audio_shift={}",
            self.num_inference_steps,
            self.video_shift,
            self.audio_shift,
        )

    def _prepare_cached_condition(self, condition):
        return self.model.prepare_text_condition(condition)

    def _encode_conditions(self, sample):
        positive = sample["conditioning"].get("positive")
        if positive is None:
            raise KeyError("MiniMax-H3 DMD requires conditioning.positive. Build the prompt cache with build_minimax_h3_prompt_latents.py and use data.train.name=latent_dataset.")
        with torch.no_grad():
            condition = self._prepare_cached_condition(positive)
        return broadcast_sequence_parallel_value(condition), None

    def _latent_shape(self, sample):
        del sample
        dmd = self.dmd_config
        height = int(dmd["height"])
        width = int(dmd["width"])
        num_frames = int(dmd["num_frames"])
        if height % 32 or width % 32:
            raise ValueError(f"MiniMax-H3 height/width must be divisible by 32, got {height}x{width}.")
        latent_frames = video_latent_num_frames(num_frames)
        latent_height = height // self.model.vae_spatial_scale_factor
        latent_width = width // self.model.vae_spatial_scale_factor
        patch_t, patch_h, patch_w = self.model.patch_size
        if latent_height % patch_h or latent_width % patch_w:
            raise ValueError(f"H3 latent canvas {latent_height}x{latent_width} is not divisible by patch {self.model.patch_size}.")
        video_rows = latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
        video_dim = self.model.video_latent_channels * patch_t * patch_h * patch_w
        audio_latents = audio_latent_num_frames(num_frames)
        audio_rows = audio_latents * 2
        return {
            "batch_size": 1,
            "num_frames": num_frames,
            "latent_frames": latent_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "audio_latents": audio_latents,
            "video_tokens": (1, video_rows, video_dim),
            "audio_tokens": (1, audio_rows, self.model.audio_latent_channels),
        }

    def sample_initial_latents(self, latent_shape):
        video = torch.randn(latent_shape["video_tokens"], device=self.model.device, dtype=self.latent_dtype)
        audio = torch.randn(latent_shape["audio_tokens"], device=self.model.device, dtype=self.latent_dtype)
        return broadcast_sequence_parallel_value((video, audio))

    def _layout(self, condition, latent_shape):
        tags = condition["text_token_tags"]
        if tags.ndim != 1:
            raise ValueError(f"text_token_tags must be one-dimensional, got {tuple(tags.shape)}.")
        if not bool((tags == 1).all()):
            raise ValueError("The initial minimax_h3_t2av_dmd trainer only supports text-only conditions.")
        key = (
            int(tags.numel()),
            latent_shape["latent_frames"],
            latent_shape["latent_height"],
            latent_shape["latent_width"],
            latent_shape["audio_latents"],
            self.model.patch_size,
            self.model.device,
        )
        layout = self._layout_cache.get(key)
        if layout is None:
            layout = build_packed_sequence(
                tags.detach().cpu(),
                latent_shape["latent_frames"],
                latent_shape["latent_height"],
                latent_shape["latent_width"],
                latent_shape["audio_latents"],
                self.model.patch_size,
            ).to(self.model.device)
            self._layout_cache[key] = layout
        return layout

    def _predict_velocity(self, model, latents, sigmas, condition, latent_shape):
        video, audio = latents
        video_sigma, audio_sigma = sigmas
        if video.shape[0] != 1 or audio.shape[0] != 1:
            raise ValueError("MiniMax-H3 packed T2AV training currently requires batch_size=1.")
        layout = self._layout(condition, latent_shape)
        timesteps, timestep_indices = build_row_timesteps(layout, video_sigma, audio_sigma)
        with model.transformer_forward_context():
            return model.denoiser_module()(
                hidden_states=video,
                audio_hidden_states=audio,
                encoder_hidden_states=condition["prompt_embeds"],
                timestep=timesteps.to(model.device),
                timestep_indices=timestep_indices.to(model.device),
                token_tags=layout.token_tags,
                position_ids=layout.position_ids,
                video_indices=layout.video_indices,
                audio_indices=layout.audio_indices,
                text_indices=layout.text_indices,
                return_dict=False,
            )

    def run_back_simulation(self, condition, latent_shape, end_step_idx, grad_enabled, xt=None):
        if xt is None:
            xt = self.sample_initial_latents(latent_shape)
        self.model.transformer.train()
        x0 = None
        for step_idx in range(end_step_idx + 1):
            video_sigma = self.video_sigmas_cpu[step_idx].to(self.model.device)
            audio_sigma = self.audio_sigmas_cpu[step_idx].to(self.model.device)
            context = torch.enable_grad if grad_enabled and step_idx == end_step_idx else torch.no_grad
            with context():
                velocity = self._predict_velocity(self.model, xt, (video_sigma, audio_sigma), condition, latent_shape)
                x0 = (
                    xt[0] + video_sigma * velocity[0],
                    xt[1] + audio_sigma * velocity[1],
                )
            next_video_sigma = self.video_sigmas_cpu[step_idx + 1].to(self.model.device)
            next_audio_sigma = self.audio_sigmas_cpu[step_idx + 1].to(self.model.device)
            xt = (
                (xt[0].float() + (video_sigma - next_video_sigma) * velocity[0].float()).to(self.latent_dtype),
                (xt[1].float() + (audio_sigma - next_audio_sigma) * velocity[1].float()).to(self.latent_dtype),
            )
        return x0[0].to(self.latent_dtype), x0[1].to(self.latent_dtype)

    def _sample_renoise_sigmas(self):
        low = float(self.dmd_config.get("renoise_sigma_min", 0.02))
        high = float(self.dmd_config.get("renoise_sigma_max", 0.98))
        if not 0.0 <= low < high <= 1.0:
            raise ValueError(f"H3 renoise sigma range must satisfy 0 <= min < max <= 1, got [{low}, {high}].")
        base = torch.empty((), device=self.model.device, dtype=torch.float32).uniform_(low, high)
        return _shift_sigma(base, self.video_shift), _shift_sigma(base, self.audio_shift)

    @staticmethod
    def _add_noise(latents, noises, sigmas):
        return (
            ((1.0 - sigmas[0]) * latents[0].float() + sigmas[0] * noises[0]).to(latents[0].dtype),
            ((1.0 - sigmas[1]) * latents[1].float() + sigmas[1] * noises[1]).to(latents[1].dtype),
        )

    def forward_loss(self, latent_shape, conditions, stage, current_iter=None, initial_noise=None):
        del current_iter
        condition, _ = conditions
        end_step_idx = self._sample_synced_int(0, self.num_inference_steps)
        generated = self.run_back_simulation(
            condition,
            latent_shape,
            end_step_idx,
            grad_enabled=stage != "fake",
            xt=initial_noise,
        )
        sigmas = self._sample_renoise_sigmas()
        noises = (torch.randn_like(generated[0], dtype=torch.float32), torch.randn_like(generated[1], dtype=torch.float32))
        renoised = self._add_noise(generated, noises, sigmas)

        if stage == "fake":
            self.fake_model.transformer.train()
            velocity_fake = self._predict_velocity(self.fake_model, renoised, sigmas, condition, latent_shape)
            # H3 predicts clean-ward velocity d x(t) / dt = x0 - noise.
            target = (generated[0].float() - noises[0], generated[1].float() - noises[1])
            return weighted_mse_pair(
                velocity_fake,
                target,
                self.video_loss_weight,
                self.audio_loss_weight,
            )

        with torch.no_grad():
            self.fake_model.transformer.eval()
            self.teacher_model.transformer.eval()
            velocity_fake = self._predict_velocity(self.fake_model, renoised, sigmas, condition, latent_shape)
            velocity_teacher = self._predict_velocity(self.teacher_model, renoised, sigmas, condition, latent_shape)
            x0_fake = (
                renoised[0] + sigmas[0] * velocity_fake[0],
                renoised[1] + sigmas[1] * velocity_fake[1],
            )
            x0_teacher = (
                renoised[0] + sigmas[0] * velocity_teacher[0],
                renoised[1] + sigmas[1] * velocity_teacher[1],
            )
        loss = dmd_loss_pair(
            generated,
            x0_fake,
            x0_teacher,
            self.video_loss_weight,
            self.audio_loss_weight,
        )
        return {"loss": loss, "dmd": loss.detach()}
