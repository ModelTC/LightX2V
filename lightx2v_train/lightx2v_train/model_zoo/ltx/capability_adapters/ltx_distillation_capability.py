"""Distribution-matching capability for LTX joint video/audio models."""

from __future__ import annotations

import torch

from lightx2v_train.model_zoo.capability_adapters.common import (
    GenericDistillationCapability,
    _negative_prompt,
    _require_single_prompt,
    _require_singleton_tensor,
)
from lightx2v_train.model_zoo.native.ltx2 import (
    AudioLatentShape,
    AudioPatchifier,
    Modality,
    SpatioTemporalScaleFactors,
    VideoLatentPatchifier,
    VideoLatentShape,
    get_pixel_coords,
)
from lightx2v_train.utils.constants import LTX2_NEGATIVE_PROMPT

from .common import LTXJointLatentShape, LTXJointLatents


class LTXDistillationCapability(GenericDistillationCapability):
    """Distribution-matching operations for joint LTX video/audio states."""

    def __init__(self, model) -> None:
        super().__init__(model)
        config = model.config["training"].get("ltx2", {})
        dmd_config = model.config["training"].get("dmd", {})
        self.video_weight = float(config.get("video_loss_weight", 1.0))
        self.audio_weight = float(config.get("audio_loss_weight", 1.0))
        self.default_fps = float(config.get("default_fps", dmd_config.get("fps", 25.0)))
        self.video_patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)

    @property
    def default_negative_prompt(self):
        return LTX2_NEGATIVE_PROMPT

    @property
    def default_lora_target_modules(self):
        return None

    def latent_shape(
        self,
        batch,
        shape_config,
        image_sizes,
        broadcast,
    ):
        del image_sizes, broadcast
        prompt = batch["conditioning"].get("prompt", "")
        _require_single_prompt(prompt)
        video_shape = shape_config.get("video_latent_shape")
        if video_shape is None:
            meta = batch["meta"]
            height = int(meta["target_height"][0].item()) if "target_height" in meta else int(shape_config.get("height", 512))
            width = int(meta["target_width"][0].item()) if "target_width" in meta else int(shape_config.get("width", 768))
            frames = int(shape_config.get("num_frames", 241))
            video_shape = (
                1,
                int(shape_config.get("video_latent_channels", 128)),
                (frames - 1) // int(shape_config.get("video_temporal_scale", 8)) + 1,
                height // int(shape_config.get("video_spatial_scale", 32)),
                width // int(shape_config.get("video_spatial_scale", 32)),
            )
        else:
            video_shape = self._singleton_shape(
                video_shape,
                "video_latent_shape",
            )

        audio_shape = shape_config.get("audio_latent_shape")
        if audio_shape is None:
            if "audio_num_frames" in shape_config:
                audio_frames = int(shape_config["audio_num_frames"])
            else:
                pixel_frames = int(
                    shape_config.get(
                        "num_frames",
                        (video_shape[2] - 1) * 8 + 1,
                    )
                )
                fps = float(shape_config.get("fps", self.default_fps))
                audio_frames = round(pixel_frames / fps * 25.0)
            audio_shape = (
                1,
                int(shape_config.get("audio_latent_channels", 8)),
                audio_frames,
                int(shape_config.get("audio_mel_bins", 16)),
            )
        else:
            audio_shape = self._singleton_shape(
                audio_shape,
                "audio_latent_shape",
            )
        return LTXJointLatentShape(
            video_latent=video_shape,
            audio_latent=audio_shape,
            video_tokens=(
                1,
                video_shape[2] * video_shape[3] * video_shape[4],
                video_shape[1],
            ),
            audio_tokens=(
                1,
                audio_shape[2],
                audio_shape[1] * audio_shape[3],
            ),
            fps=float(shape_config.get("fps", self.default_fps)),
        )

    def encode_conditions(
        self,
        batch,
        negative_prompt,
        guidance_scale,
        broadcast,
    ):
        conditioning = batch["conditioning"]
        positive_value = conditioning.get("positive")
        if positive_value is None:
            return super().encode_conditions(
                batch,
                negative_prompt,
                guidance_scale,
                broadcast,
            )
        with torch.no_grad():
            positive = self._cached_condition(positive_value)
            if guidance_scale > 1:
                negative_value = conditioning.get("negative")
                if negative_value is not None:
                    negative = self._cached_condition(negative_value)
                else:
                    prompt = conditioning.get("prompt", "")
                    scalar = _require_single_prompt(prompt)
                    negative = self.model.encode_prompt_condition(
                        _negative_prompt(
                            conditioning,
                            negative_prompt,
                            scalar=scalar,
                        )
                    )
            else:
                negative = None
        return (
            broadcast(positive),
            broadcast(negative) if negative is not None else None,
        )

    def _cached_condition(self, value):
        if isinstance(value, (list, tuple)) and len(value) == 3:
            video, audio, mask = value
        elif isinstance(value, dict) and {
            "video_context",
            "audio_context",
            "context_mask",
        }.issubset(value):
            video = value["video_context"]
            audio = value["audio_context"]
            mask = value["context_mask"]
        else:
            return self.model.prepare_text_condition(value)
        video = _require_singleton_tensor(
            video.to(device=self.device, dtype=self.model.running_dtype),
            "LTX video context",
        )
        audio = _require_singleton_tensor(
            audio.to(device=self.device, dtype=self.model.running_dtype),
            "LTX audio context",
        )
        mask = _require_singleton_tensor(
            mask.to(device=self.device),
            "LTX context mask",
        )
        return video, audio, mask

    def predict_velocity(self, latents, sigma, condition):
        video_context, audio_context, context_mask = condition
        _require_singleton_tensor(latents.video, "LTX video latent")
        _require_singleton_tensor(latents.audio, "LTX audio latent")
        sigma = sigma.to(device=self.device, dtype=self.model.running_dtype)
        video_timesteps = (
            sigma.view(-1, 1)
            .expand(
                1,
                latents.video.shape[1],
            )
            .clone()
        )
        audio_timesteps = (
            sigma.view(-1, 1)
            .expand(
                1,
                latents.audio.shape[1],
            )
            .clone()
        )
        video = Modality(
            enabled=True,
            latent=latents.video.to(
                device=self.device,
                dtype=self.model.running_dtype,
            ),
            sigma=sigma,
            timesteps=video_timesteps,
            positions=self._video_token_positions(latents.shape),
            context=video_context.to(
                device=self.device,
                dtype=self.model.running_dtype,
            ),
            context_mask=context_mask.to(device=self.device),
        )
        audio = Modality(
            enabled=True,
            latent=latents.audio.to(
                device=self.device,
                dtype=self.model.running_dtype,
            ),
            sigma=sigma,
            timesteps=audio_timesteps,
            positions=self._audio_token_positions(latents.shape),
            context=audio_context.to(
                device=self.device,
                dtype=self.model.running_dtype,
            ),
            context_mask=context_mask.to(device=self.device),
        )
        with self.model.transformer_forward_context():
            video_prediction, audio_prediction = self.model.denoiser_module()(
                video=video,
                audio=audio,
                perturbations=None,
            )
        return LTXJointLatents(
            video_prediction,
            audio_prediction,
            latents.shape,
        )

    def predict_guided_velocity(
        self,
        latents,
        sigma,
        condition,
        negative_condition,
        guidance_scale,
        cfg_norm,
    ):
        positive = self.predict_velocity(latents, sigma, condition)
        if negative_condition is None:
            return positive
        negative = self.predict_velocity(latents, sigma, negative_condition)
        return LTXJointLatents(
            self._cfg(
                positive.video,
                negative.video,
                guidance_scale,
                cfg_norm,
            ),
            self._cfg(
                positive.audio,
                negative.audio,
                guidance_scale,
                cfg_norm,
            ),
            latents.shape,
        )

    def initial_latents(self, latent_shape, dtype, broadcast):
        del broadcast
        if latent_shape.video_tokens[0] != 1 or latent_shape.audio_tokens[0] != 1:
            raise ValueError(f"LTX DMD latent shape must start with 1, got {latent_shape}.")
        return LTXJointLatents(
            torch.randn(
                latent_shape.video_tokens,
                device=self.device,
                dtype=dtype,
            ),
            torch.randn(
                latent_shape.audio_tokens,
                device=self.device,
                dtype=dtype,
            ),
            latent_shape,
        )

    @staticmethod
    def latent_hw(latent_shape):
        return None

    @staticmethod
    def random_noise_like(latents, dtype, broadcast):
        del broadcast
        return LTXJointLatents(
            torch.randn_like(latents.video, dtype=dtype),
            torch.randn_like(latents.audio, dtype=dtype),
            latents.shape,
        )

    @staticmethod
    def add_noise(scheduler, latents, noise, sigma):
        return LTXJointLatents(
            scheduler.add_noise(latents.video, noise.video, sigma).to(latents.video.dtype),
            scheduler.add_noise(latents.audio, noise.audio, sigma).to(latents.audio.dtype),
            latents.shape,
        )

    @staticmethod
    def training_target(latents, noise):
        return LTXJointLatents(
            noise.video - latents.video.float(),
            noise.audio - latents.audio.float(),
            latents.shape,
        )

    @staticmethod
    def step(scheduler, velocity, step_index, sample):
        video_next, video_x0 = scheduler.step_by_index(
            velocity.video,
            step_index,
            sample.video,
        )
        audio_next, audio_x0 = scheduler.step_by_index(
            velocity.audio,
            step_index,
            sample.audio,
        )
        return (
            LTXJointLatents(video_next, audio_next, sample.shape),
            LTXJointLatents(video_x0, audio_x0, sample.shape),
        )

    @staticmethod
    def x0_from_velocity(sample, velocity, sigma):
        def project(value, flow):
            expanded = sigma.reshape(
                sigma.shape[0],
                *([1] * (value.ndim - 1)),
            )
            return value - expanded * flow

        return LTXJointLatents(
            project(sample.video, velocity.video),
            project(sample.audio, velocity.audio),
            sample.shape,
        )

    def regression_loss(self, prediction, target):
        video = torch.nn.functional.mse_loss(
            prediction.video.float(),
            target.video.float(),
        )
        audio = torch.nn.functional.mse_loss(
            prediction.audio.float(),
            target.audio.float(),
        )
        return self.video_weight * video + self.audio_weight * audio

    def dmd_loss(self, latents, fake_x0, teacher_x0):
        return self.video_weight * super().dmd_loss(
            latents.video,
            fake_x0.video,
            teacher_x0.video,
        ) + self.audio_weight * super().dmd_loss(
            latents.audio,
            fake_x0.audio,
            teacher_x0.audio,
        )

    @staticmethod
    def detach(value):
        return LTXJointLatents(
            value.video.detach(),
            value.audio.detach(),
            value.shape,
        )

    @staticmethod
    def to_dtype(value, dtype):
        return LTXJointLatents(
            value.video.to(dtype=dtype),
            value.audio.to(dtype=dtype),
            value.shape,
        )

    def extract_real_latents(self, batch, dtype, broadcast):
        raise ValueError("LTX joint DMD does not support real-data fake loss.")

    def _video_token_positions(self, shape):
        _, channels, frames, height, width = shape.video_latent
        latent_coords = self.video_patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                frames=frames,
                height=height,
                width=width,
                batch=shape.video_latent[0],
                channels=channels,
            ),
            device=self.device,
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=SpatioTemporalScaleFactors.default(),
            causal_fix=True,
        ).float()
        positions[:, 0, ...] /= shape.fps
        return positions

    def _audio_token_positions(self, shape):
        _, channels, frames, mel_bins = shape.audio_latent
        return self.audio_patchifier.get_patch_grid_bounds(
            output_shape=AudioLatentShape(
                frames=frames,
                mel_bins=mel_bins,
                batch=shape.audio_latent[0],
                channels=channels,
            ),
            device=self.device,
        )

    @staticmethod
    def _singleton_shape(shape, name):
        shape = tuple(int(dimension) for dimension in shape)
        expected = 4 if name.startswith("audio") else 5
        without_batch = expected - 1
        if len(shape) == without_batch:
            return (1, *shape)
        if len(shape) == expected:
            if shape[0] != 1:
                raise ValueError(f"training.dmd.{name} must begin with singleton dimension 1, got {shape}.")
            return shape
        raise ValueError(f"training.dmd.{name} has unsupported shape {shape}.")
