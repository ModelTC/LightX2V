import math
import os

import torch
from loguru import logger

from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value
from lightx2v_train.utils.constants import LTX2_NEGATIVE_PROMPT
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .math import (
    detach_pair,
    dmd_loss_pair,
    to_dtype_pair,
    velocity_to_x0,
    weighted_mse_pair,
)
from .video_ar_trainer import VideoArDmdTrainer
from .video_trainer import VideoDmdTrainer


class _LTX2T2AVDmdMixin:
    trainer_name = "ltx_t2av_dmd"
    allowed_model_names = {"ltx_t2av"}
    default_negative_prompt = LTX2_NEGATIVE_PROMPT
    default_lora_target_modules = None
    supports_diversity_loss = False
    supports_real_data_fake = False

    def __init__(self, config):
        from lightx2v_train.model_zoo.native.ltx2 import (
            AudioLatentShape,
            AudioPatchifier,
            Modality,
            SpatioTemporalScaleFactors,
            VideoLatentPatchifier,
            VideoLatentShape,
            get_pixel_coords,
        )

        self._ltx_audio_latent_shape = AudioLatentShape
        self._ltx_modality = Modality
        self._ltx_scale_factors = SpatioTemporalScaleFactors
        self._ltx_video_latent_shape = VideoLatentShape
        self._ltx_get_pixel_coords = get_pixel_coords
        super().__init__(config)
        model_name = self.model_config["name"]
        if model_name not in self.allowed_model_names:
            allowed = ", ".join(repr(name) for name in sorted(self.allowed_model_names))
            raise ValueError(f"{self.trainer_name} trainer requires model.name in {{{allowed}}}.")

        ltx2_config = self.training_config.get("ltx2", {})
        self.video_loss_weight = float(ltx2_config.get("video_loss_weight", 1.0))
        self.audio_loss_weight = float(ltx2_config.get("audio_loss_weight", 1.0))
        self.default_fps = float(ltx2_config.get("default_fps", self.dmd_config.get("fps", 25.0)))
        self.video_patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)
        self.denoising_sigma_values = self._validate_denoising_sigma_values(self.dmd_config.get("denoising_sigma_values"))
        self._logged_denoising_sigma_values = False

        if self.cdm_enabled:
            raise ValueError("ltx_t2av_dmd does not support training.dmd.cdm yet.")

    def _load_student_checkpoint(self, checkpoint_path, strict=True):
        loaded_path = getattr(self.model, "loaded_student_checkpoint_path", None)
        requested_path = os.path.abspath(os.path.expanduser(str(checkpoint_path)))
        if loaded_path == requested_path:
            logger.info("[train] LTX2 AR student checkpoint was loaded during model construction: {}", requested_path)
            return
        super()._load_student_checkpoint(checkpoint_path, strict=strict)

    def _encode_conditions(self, sample):
        conditioning = sample["conditioning"]
        positive = conditioning.get("positive")
        if positive is None:
            return super()._encode_conditions(sample)

        with torch.no_grad():
            condition = self._prepare_cached_condition(positive)
            if self.guidance_scale > 1:
                negative = conditioning.get("negative")
                if negative is not None:
                    negative_condition = self._prepare_cached_condition(negative)
                else:
                    prompt = conditioning.get("prompt", "")
                    is_scalar_prompt = isinstance(prompt, str)
                    batch_size = 1 if is_scalar_prompt else len(prompt)
                    negative_prompt = self._negative_prompt_for_conditioning(
                        conditioning,
                        batch_size,
                        return_scalar=is_scalar_prompt,
                    )
                    try:
                        negative_condition = self.model.encode_prompt_condition(negative_prompt)
                    except RuntimeError as exc:
                        raise RuntimeError(
                            "LTX DMD CFG needs negative conditions. Provide data.train.negative_condition_path or keep model.load_text_encoder=true for online negative prompt encoding."
                        ) from exc
            else:
                negative_condition = None

        condition = broadcast_sequence_parallel_value(condition)
        negative_condition = broadcast_sequence_parallel_value(negative_condition) if negative_condition is not None else None
        return condition, negative_condition

    def _prepare_cached_condition(self, conditions):
        if isinstance(conditions, (list, tuple)) and len(conditions) == 3:
            video_context, audio_context, context_mask = conditions
            return (
                video_context.to(device=self.model.device, dtype=self.running_dtype),
                audio_context.to(device=self.model.device, dtype=self.running_dtype),
                context_mask.to(device=self.model.device),
            )

        if isinstance(conditions, dict) and {"video_context", "audio_context", "context_mask"}.issubset(conditions):
            return (
                conditions["video_context"].to(device=self.model.device, dtype=self.running_dtype),
                conditions["audio_context"].to(device=self.model.device, dtype=self.running_dtype),
                conditions["context_mask"].to(device=self.model.device),
            )

        return self.model.prepare_text_condition(conditions)

    def _latent_shape(self, sample):
        prompt = sample["conditioning"].get("prompt", "")
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        dmd_config = self.dmd_config

        video_shape = dmd_config.get("video_latent_shape")
        if video_shape is not None:
            video_shape = self._shape_with_batch(video_shape, batch_size, "video_latent_shape")
        else:
            meta = sample["meta"]
            height = int(meta["target_height"][0].item()) if "target_height" in meta else int(dmd_config.get("height", 512))
            width = int(meta["target_width"][0].item()) if "target_width" in meta else int(dmd_config.get("width", 768))
            num_frames = int(dmd_config.get("num_frames", 241))
            latent_frames = (num_frames - 1) // int(dmd_config.get("video_temporal_scale", 8)) + 1
            latent_height = int(height) // int(dmd_config.get("video_spatial_scale", 32))
            latent_width = int(width) // int(dmd_config.get("video_spatial_scale", 32))
            video_shape = (
                batch_size,
                int(dmd_config.get("video_latent_channels", 128)),
                latent_frames,
                latent_height,
                latent_width,
            )

        audio_shape = dmd_config.get("audio_latent_shape")
        if audio_shape is not None:
            audio_shape = self._shape_with_batch(audio_shape, batch_size, "audio_latent_shape")
        else:
            if "audio_num_frames" in dmd_config:
                audio_frames = int(dmd_config["audio_num_frames"])
            else:
                pixel_frames = int(dmd_config.get("num_frames", (video_shape[2] - 1) * 8 + 1))
                fps = float(dmd_config.get("fps", self.default_fps))
                audio_frames = round((pixel_frames / fps) * 25.0)
            audio_shape = (
                batch_size,
                int(dmd_config.get("audio_latent_channels", 8)),
                audio_frames,
                int(dmd_config.get("audio_mel_bins", 16)),
            )

        video_tokens = video_shape[2] * video_shape[3] * video_shape[4]
        audio_tokens = audio_shape[2]
        return {
            "video_latent": video_shape,
            "audio_latent": audio_shape,
            "video_tokens": (batch_size, video_tokens, video_shape[1]),
            "audio_tokens": (batch_size, audio_tokens, audio_shape[1] * audio_shape[3]),
            "fps": float(dmd_config.get("fps", self.default_fps)),
        }

    def _prepare_sampling_schedule(self, latent_shape):
        if self.random_schedule_enabled:
            num_steps = self._sample_synced_int(self.random_schedule_num_steps_min, self.random_schedule_num_steps_max + 1)
            self.scheduler.set_random_timesteps(
                self.random_schedule_num_steps_min,
                self.random_schedule_num_steps_max,
                sigma_min=self.random_schedule_sigma_min,
                sigma_max=self.random_schedule_sigma_max,
                sampling_method=self.random_schedule_sampling_method,
                device=self.model.device,
                num_steps=num_steps,
            )
            return
        sigmas = None if self.denoising_sigma_values is None else self.denoising_sigma_values[:-1]
        self.scheduler.set_timesteps(
            self.num_inference_steps,
            sigmas=sigmas,
            device=self.model.device,
        )
        if not self._logged_denoising_sigma_values:
            logger.info(
                "[train] {} denoising_sigma_values={}",
                self.trainer_name,
                [round(float(value), 6) for value in self.scheduler.sigmas.detach().cpu()],
            )
            self._logged_denoising_sigma_values = True

    def _validate_denoising_sigma_values(self, values):
        if values is None:
            return None
        values = tuple(float(value) for value in values)
        expected = self.num_inference_steps + 1
        if len(values) != expected:
            raise ValueError(f"training.dmd.denoising_sigma_values must contain num_inference_steps + 1 values ({expected}), got {len(values)}.")
        if abs(values[0] - 1.0) > 1e-6 or abs(values[-1]) > 1e-6:
            raise ValueError("training.dmd.denoising_sigma_values must start at 1.0 and end at 0.0.")
        if any(current <= following for current, following in zip(values, values[1:])):
            raise ValueError("training.dmd.denoising_sigma_values must be strictly decreasing.")
        return values

    def sample_initial_latents(self, latent_shape):
        video = torch.randn(latent_shape["video_tokens"], device=self.model.device, dtype=self.running_dtype)
        audio = torch.randn(latent_shape["audio_tokens"], device=self.model.device, dtype=self.running_dtype)
        return video, audio

    def run_back_simulation(self, condition, latent_shape, end_step_idx, grad_enabled, xt=None):
        if xt is None:
            xt = self.sample_initial_latents(latent_shape)
        x0 = None
        xt_end = None
        vt_end = None
        self.model.transformer.train()
        for idx in range(end_step_idx + 1):
            sigma = self.scheduler.sigma_at(idx, latent_shape["video_tokens"][0], device=self.model.device, dtype=self.running_dtype)
            context = torch.enable_grad if (grad_enabled and idx == end_step_idx) else torch.no_grad
            with context():
                velocity = self._predict_velocity(self.model, xt, sigma, condition, latent_shape)
            if idx == end_step_idx:
                xt_end = self._detach_pair(xt)
                vt_end = self._detach_pair(velocity)
            xt, x0 = self._step_pair_by_index(velocity, idx, xt)
        return self._to_dtype_pair(x0, self.running_dtype), xt_end, vt_end

    def forward_loss(
        self,
        latent_shape,
        conditions,
        stage,
        current_iter=None,
        initial_noise=None,
    ):
        condition, negative_condition = conditions
        self._prepare_sampling_schedule(latent_shape)
        end_step_idx = self.sample_end_step()
        xt_start = self.sample_initial_latents(latent_shape) if initial_noise is None else initial_noise
        x0, xt_end, vt_end = self.run_back_simulation(condition, latent_shape, end_step_idx, grad_enabled=(stage != "fake"), xt=xt_start)

        batch_size = latent_shape["video_tokens"][0]
        sigma = self.scheduler.sample_renoise_sigma(batch_size, device=self.model.device, dtype=self.running_dtype)
        noise = (
            torch.randn_like(x0[0], dtype=torch.float32),
            torch.randn_like(x0[1], dtype=torch.float32),
        )
        renoised_xt = self._add_noise_pair(x0, noise, sigma)

        if stage == "fake":
            self.fake_model.transformer.train()
            velocity_fake = self._predict_velocity(self.fake_model, renoised_xt, sigma, condition, latent_shape)
            velocity_gt = (noise[0] - x0[0].float(), noise[1] - x0[1].float())
            return self._weighted_mse_pair(velocity_fake, velocity_gt)

        with torch.no_grad():
            self.fake_model.transformer.eval()
            self.teacher_model.transformer.eval()
            velocity_fake = self._predict_velocity(self.fake_model, renoised_xt, sigma, condition, latent_shape)
            velocity_teacher_cond = self._predict_velocity(self.teacher_model, renoised_xt, sigma, condition, latent_shape)
            if negative_condition is None or self.guidance_scale == 0:
                velocity_teacher = velocity_teacher_cond
            else:
                velocity_teacher_uncond = self._predict_velocity(self.teacher_model, renoised_xt, sigma, negative_condition, latent_shape)
                velocity_teacher = self._cfg_pair(velocity_teacher_cond, velocity_teacher_uncond)

        x_pred_fake = self._x0_from_velocity(renoised_xt, velocity_fake, sigma)
        x_pred_teacher = self._x0_from_velocity(renoised_xt, velocity_teacher, sigma)
        return self._dmd_loss_pair(x0, x_pred_fake, x_pred_teacher)

    def _predict_velocity(self, model, latents, sigma, condition, latent_shape):
        video_context, audio_context, context_mask = condition
        video_tokens, audio_tokens = latents
        batch_size = video_tokens.shape[0]
        sigma = sigma.to(device=self.model.device, dtype=self.running_dtype)
        timesteps_video = sigma.view(-1, 1).expand(batch_size, video_tokens.shape[1]).clone()
        timesteps_audio = sigma.view(-1, 1).expand(batch_size, audio_tokens.shape[1]).clone()
        video_positions = self._get_video_positions(latent_shape, batch_size, video_tokens.device)
        audio_positions = self._get_audio_positions(latent_shape, batch_size, audio_tokens.device)

        video_modality = self._ltx_modality(
            enabled=True,
            latent=video_tokens.to(device=model.device, dtype=self.running_dtype),
            sigma=sigma,
            timesteps=timesteps_video,
            positions=video_positions,
            context=video_context.to(device=model.device, dtype=self.running_dtype),
            context_mask=context_mask.to(device=model.device),
        )
        audio_modality = self._ltx_modality(
            enabled=True,
            latent=audio_tokens.to(device=model.device, dtype=self.running_dtype),
            sigma=sigma,
            timesteps=timesteps_audio,
            positions=audio_positions,
            context=audio_context.to(device=model.device, dtype=self.running_dtype),
            context_mask=context_mask.to(device=model.device),
        )
        with model.transformer_forward_context():
            return model.denoiser_module()(
                video=video_modality,
                audio=audio_modality,
                perturbations=None,
            )

    def _get_video_positions(self, latent_shape, batch_size, device):
        _, channels, frames, height, width = latent_shape["video_latent"]
        latent_coords = self.video_patchifier.get_patch_grid_bounds(
            output_shape=self._ltx_video_latent_shape(
                frames=frames,
                height=height,
                width=width,
                batch=batch_size,
                channels=channels,
            ),
            device=device,
        )
        pixel_coords = self._ltx_get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=self._ltx_scale_factors.default(),
            causal_fix=True,
        ).float()
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / latent_shape["fps"]
        return pixel_coords

    def _get_audio_positions(self, latent_shape, batch_size, device):
        _, channels, frames, mel_bins = latent_shape["audio_latent"]
        return self.audio_patchifier.get_patch_grid_bounds(
            output_shape=self._ltx_audio_latent_shape(
                frames=frames,
                mel_bins=mel_bins,
                batch=batch_size,
                channels=channels,
            ),
            device=device,
        )

    def _add_noise_pair(self, latents, noises, sigma):
        return (
            self.scheduler.add_noise(latents[0], noises[0], sigma).to(dtype=latents[0].dtype),
            self.scheduler.add_noise(latents[1], noises[1], sigma).to(dtype=latents[1].dtype),
        )

    def _step_pair_by_index(self, velocity, step_idx, sample):
        video_next, video_x0 = self.scheduler.step_by_index(velocity[0], step_idx, sample[0])
        audio_next, audio_x0 = self.scheduler.step_by_index(velocity[1], step_idx, sample[1])
        return (video_next, audio_next), (video_x0, audio_x0)

    def _x0_from_velocity(self, xt, velocity, sigma):
        return (
            velocity_to_x0(xt[0], velocity[0], sigma),
            velocity_to_x0(xt[1], velocity[1], sigma),
        )

    def _cfg_pair(self, cond, uncond):
        return (
            self._do_cfg(cond[0], uncond[0], self.guidance_scale, self.cfg_norm),
            self._do_cfg(cond[1], uncond[1], self.guidance_scale, self.cfg_norm),
        )

    def _dmd_loss_pair(self, latents, x_pred_fake_flow, x_pred_teacher):
        return dmd_loss_pair(
            latents,
            x_pred_fake_flow,
            x_pred_teacher,
            self.video_loss_weight,
            self.audio_loss_weight,
        )

    def _weighted_mse_pair(self, pred, target):
        return weighted_mse_pair(
            pred,
            target,
            self.video_loss_weight,
            self.audio_loss_weight,
        )

    @staticmethod
    def _detach_pair(value):
        return detach_pair(value)

    @staticmethod
    def _to_dtype_pair(value, dtype):
        return to_dtype_pair(value, dtype)

    @staticmethod
    def _shape_with_batch(shape, batch_size, name):
        shape = tuple(int(dim) for dim in shape)
        if name.startswith("audio") and len(shape) == 3:
            return (batch_size, *shape)
        if name.startswith("audio") and len(shape) == 4:
            if shape[0] != batch_size:
                return (batch_size, *shape[1:])
            return shape
        if name.startswith("video") and len(shape) == 4:
            return (batch_size, *shape)
        if name.startswith("video") and len(shape) == 5:
            if shape[0] != batch_size:
                return (batch_size, *shape[1:])
            return shape
        raise ValueError(f"training.dmd.{name} has unsupported shape {shape}.")


@TRAINER_REGISTER("ltx_t2av_dmd")
class LTX2T2AVDmdTrainer(_LTX2T2AVDmdMixin, VideoDmdTrainer):
    trainer_name = "ltx_t2av_dmd"


@TRAINER_REGISTER("ltx_t2av_ar_dmd")
class LTX2T2AVArDmdTrainer(_LTX2T2AVDmdMixin, VideoArDmdTrainer):
    trainer_name = "ltx_t2av_ar_dmd"
    allowed_model_names = {"ltx_t2av_ar"}

    def __init__(self, config):
        super().__init__(config)
        if self.denoising_sigma_values is None:
            raise ValueError("ltx_t2av_ar_dmd requires training.dmd.denoising_sigma_values so training and inference use the same distilled trajectory.")

    def run_back_simulation(self, condition, latent_shape, end_step_idx, grad_enabled, xt=None):
        transformer = self.model.denoiser_module()
        if not self._ltx_transformer_has_inference_cache(transformer):
            raise RuntimeError("ltx_t2av_ar_dmd requires an LTX transformer with _forward_inference/KV cache support.")

        if xt is None:
            xt = self.sample_initial_latents(latent_shape)
        video_tokens, audio_tokens = xt
        batch_size = video_tokens.shape[0]
        _, _, num_frames, latent_height, latent_width = latent_shape["video_latent"]
        frame_seq_length = int(latent_height) * int(latent_width)
        if video_tokens.shape[1] != num_frames * frame_seq_length:
            if video_tokens.shape[1] % num_frames != 0:
                raise ValueError(f"LTX AR-DMD video token length {video_tokens.shape[1]} is not divisible by latent frames {num_frames}.")
            frame_seq_length = video_tokens.shape[1] // num_frames
        if num_frames % self.num_frame_per_chunk != 0:
            raise ValueError(f"LTX AR-DMD latent frames={num_frames} must be divisible by num_frame_per_chunk={self.num_frame_per_chunk}.")

        self.model.transformer.train()
        num_blocks = num_frames // self.num_frame_per_chunk
        video_positions = self._get_video_positions(latent_shape, batch_size, video_tokens.device)
        audio_positions = self._get_audio_positions(latent_shape, batch_size, audio_tokens.device)
        kv_cache = self._new_ltx_kv_caches(
            batch_size=batch_size,
            dtype=video_tokens.dtype,
            device=video_tokens.device,
            video_total_tokens=video_tokens.shape[1],
            audio_total_tokens=audio_tokens.shape[1],
            video_frame_seq_length=frame_seq_length,
            num_frames=num_frames,
        )

        output_video_chunks = []
        output_audio_chunks = []
        for block_idx in range(num_blocks):
            frame_start = block_idx * self.num_frame_per_chunk
            frame_end = frame_start + self.num_frame_per_chunk
            video_start = frame_start * frame_seq_length
            video_end = frame_end * frame_seq_length
            audio_start, audio_end = self._audio_token_range(block_idx, num_blocks, audio_tokens.shape[1])
            latents = (
                video_tokens[:, video_start:video_end],
                audio_tokens[:, audio_start:audio_end],
            )
            pos = (
                video_positions[:, :, video_start:video_end],
                audio_positions[:, :, audio_start:audio_end],
            )

            x0 = None
            for idx in range(end_step_idx + 1):
                sigma = self.scheduler.sigma_at(idx, batch_size, device=self.model.device, dtype=self.running_dtype)
                context = torch.enable_grad if (grad_enabled and idx == end_step_idx) else torch.no_grad
                with context():
                    velocity = self._predict_velocity_chunk(
                        self.model,
                        latents,
                        sigma,
                        condition,
                        pos,
                        kv_cache=kv_cache,
                        video_current_start=video_start,
                        audio_current_start=audio_start,
                    )
                    x0 = self._x0_from_velocity(latents, velocity, sigma)

                if idx < end_step_idx:
                    next_sigma = self.scheduler.sigma_at(
                        idx + 1,
                        batch_size,
                        device=self.model.device,
                        dtype=self.running_dtype,
                    )
                    with torch.no_grad():
                        noise = (
                            torch.randn_like(x0[0]),
                            torch.randn_like(x0[1]),
                        )
                        latents = self._add_noise_pair(
                            x0,
                            noise,
                            next_sigma,
                        )

            output_video_chunks.append(x0[0])
            output_audio_chunks.append(x0[1])

            cache_latents = self._detach_pair(x0)
            if self.context_noise > 0:
                cache_sigma = torch.full((batch_size,), self.context_noise, device=self.model.device, dtype=self.running_dtype)
                cache_noise = (torch.randn_like(cache_latents[0]), torch.randn_like(cache_latents[1]))
                cache_latents = self._add_noise_pair(cache_latents, cache_noise, cache_sigma)
            else:
                cache_sigma = torch.zeros((batch_size,), device=self.model.device, dtype=self.running_dtype)
            with torch.no_grad():
                self._predict_velocity_chunk(
                    self.model,
                    cache_latents,
                    cache_sigma,
                    condition,
                    pos,
                    kv_cache=kv_cache,
                    video_current_start=video_start,
                    audio_current_start=audio_start,
                )

        return (
            self._to_dtype_pair((torch.cat(output_video_chunks, dim=1), torch.cat(output_audio_chunks, dim=1)), self.running_dtype),
            None,
            None,
        )

    def _predict_velocity_chunk(
        self,
        model,
        latents,
        sigma,
        condition,
        positions,
        *,
        kv_cache=None,
        video_current_start=0,
        audio_current_start=0,
    ):
        video_context, audio_context, context_mask = condition
        video_tokens, audio_tokens = latents
        video_positions, audio_positions = positions
        batch_size = video_tokens.shape[0]
        sigma = sigma.to(device=model.device, dtype=self.running_dtype)
        video_modality = self._ltx_modality(
            enabled=True,
            latent=video_tokens.to(device=model.device, dtype=self.running_dtype),
            sigma=sigma,
            timesteps=sigma.view(-1, 1).expand(batch_size, video_tokens.shape[1]).clone(),
            positions=video_positions.to(device=model.device),
            context=video_context.to(device=model.device, dtype=self.running_dtype),
            context_mask=context_mask.to(device=model.device),
        )
        audio_modality = self._ltx_modality(
            enabled=True,
            latent=audio_tokens.to(device=model.device, dtype=self.running_dtype),
            sigma=sigma,
            timesteps=sigma.view(-1, 1).expand(batch_size, audio_tokens.shape[1]).clone(),
            positions=audio_positions.to(device=model.device),
            context=audio_context.to(device=model.device, dtype=self.running_dtype),
            context_mask=context_mask.to(device=model.device),
        )
        with model.transformer_forward_context():
            return model.denoiser_module()(
                video=video_modality,
                audio=audio_modality,
                perturbations=None,
                kv_cache=kv_cache,
                video_current_start=video_current_start,
                audio_current_start=audio_current_start,
            )

    def _new_ltx_kv_caches(
        self,
        batch_size,
        dtype,
        device,
        video_total_tokens,
        audio_total_tokens,
        video_frame_seq_length,
        num_frames,
    ):
        transformer = self._unwrap_ltx_transformer(self.model.denoiser_module())
        num_layers = len(transformer.transformer_blocks)
        video_heads = int(transformer.num_attention_heads)
        video_head_dim = int(transformer.inner_dim // transformer.num_attention_heads)
        audio_heads = int(transformer.audio_num_attention_heads)
        audio_head_dim = int(transformer.audio_inner_dim // transformer.audio_num_attention_heads)
        local_attn_size = int(getattr(transformer, "local_attn_size", -1))
        sink_size = int(getattr(transformer, "sink_size", 0))
        video_chunk_tokens = self.num_frame_per_chunk * video_frame_seq_length
        if local_attn_size == -1:
            video_cache_size = int(video_total_tokens)
            video_attention_window = int(video_total_tokens)
        else:
            video_attention_window = max(video_chunk_tokens, local_attn_size * video_frame_seq_length)
            video_cache_size = max(video_attention_window, sink_size * video_frame_seq_length + video_chunk_tokens)

        audio_tokens_per_frame = max(1, math.ceil(audio_total_tokens / max(1, num_frames)))
        audio_chunk_tokens = max(1, math.ceil(audio_total_tokens / (num_frames // self.num_frame_per_chunk)))
        if local_attn_size == -1:
            audio_cache_size = int(audio_total_tokens)
            audio_attention_window = int(audio_total_tokens)
        else:
            audio_attention_window = max(audio_chunk_tokens, local_attn_size * audio_tokens_per_frame)
            audio_cache_size = max(audio_attention_window, sink_size * audio_tokens_per_frame + audio_chunk_tokens)

        cache = {"video": [], "audio": []}
        for _ in range(num_layers):
            cache["video"].append(
                self._empty_ltx_cache(
                    batch_size,
                    video_cache_size,
                    video_heads,
                    video_head_dim,
                    dtype,
                    device,
                    attention_window_size=video_attention_window,
                    sink_tokens=sink_size * video_frame_seq_length,
                )
            )
            cache["audio"].append(
                self._empty_ltx_cache(
                    batch_size,
                    audio_cache_size,
                    audio_heads,
                    audio_head_dim,
                    dtype,
                    device,
                    attention_window_size=audio_attention_window,
                    sink_tokens=sink_size * audio_tokens_per_frame,
                )
            )
        return cache

    @staticmethod
    def _empty_ltx_cache(batch_size, cache_size, heads, head_dim, dtype, device, *, attention_window_size, sink_tokens):
        return {
            "k": torch.zeros((batch_size, cache_size, heads, head_dim), dtype=dtype, device=device),
            "v": torch.zeros((batch_size, cache_size, heads, head_dim), dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "attention_window_size": int(attention_window_size),
            "sink_tokens": int(sink_tokens),
        }

    @staticmethod
    def _audio_token_range(block_idx, num_blocks, audio_total_tokens):
        start = (block_idx * audio_total_tokens + num_blocks - 1) // num_blocks
        end = ((block_idx + 1) * audio_total_tokens + num_blocks - 1) // num_blocks
        return start, end

    @staticmethod
    def _ltx_transformer_has_inference_cache(transformer):
        return LTX2T2AVArDmdTrainer._unwrap_ltx_transformer(transformer) is not None

    @staticmethod
    def _unwrap_ltx_transformer(transformer):
        for candidate in (transformer, getattr(transformer, "module", None), getattr(transformer, "_fsdp_wrapped_module", None)):
            if candidate is not None and hasattr(candidate, "_forward_inference"):
                return candidate
        return None
