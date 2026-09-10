"""Distribution-matching capability for MiniMax-H3 T2AV."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lightx2v_train.model_capabilities import DistributionMatchingProfile
from lightx2v_train.model_zoo.capability_adapters.common import (
    GenericDistributionMatchingCapability,
    _cached_condition,
    _require_single_prompt,
    _require_singleton_tensor,
)
from lightx2v_train.model_zoo.native.minimax_h3 import (
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    video_latent_num_frames,
)
from lightx2v_train.utils.generation_shapes import resolve_generation_shape

from .common import MiniMaxH3JointLatents, MiniMaxH3LatentShape


@dataclass(frozen=True)
class MiniMaxH3DistributionMatchingOptions:
    video_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0
    video_flow_shift: float = 6.0
    audio_flow_shift: float = 3.0

    @classmethod
    def from_mapping(cls, config: Mapping | None) -> "MiniMaxH3DistributionMatchingOptions":
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise ValueError("model.capabilities.distribution_matching must be a mapping.")
        options = cls(
            video_loss_weight=float(config.get("video_loss_weight", 1.0)),
            audio_loss_weight=float(config.get("audio_loss_weight", 1.0)),
            video_flow_shift=float(config.get("video_flow_shift", 6.0)),
            audio_flow_shift=float(config.get("audio_flow_shift", 3.0)),
        )
        if options.video_flow_shift <= 0 or options.audio_flow_shift <= 0:
            raise ValueError("MiniMax-H3 video_flow_shift and audio_flow_shift must be positive.")
        if options.video_loss_weight < 0 or options.audio_loss_weight < 0:
            raise ValueError("MiniMax-H3 video_loss_weight and audio_loss_weight cannot be negative.")
        if options.video_loss_weight == 0 and options.audio_loss_weight == 0:
            raise ValueError("At least one MiniMax-H3 modality loss weight must be non-zero.")
        return options


def _shift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply H3's rational flow shift to an unshifted noise level."""

    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def _expand_sigma(sigma: torch.Tensor, ndim: int) -> torch.Tensor:
    if sigma.ndim == 0:
        sigma = sigma.reshape(1)
    return sigma.reshape(sigma.shape[0], *([1] * (ndim - 1)))


class MiniMaxH3DistributionMatchingCapability(GenericDistributionMatchingCapability):
    """H3-specific operations consumed by the framework's generic DMD loop.

    H3 jointly denoises packed video and stereo-audio tokens. It also uses a
    clean-ward velocity (``x0 - noise``), while most LightX2V models use the
    opposite flow direction. Keeping those details here lets ``DmdTrainer``
    manage roles, rollout, optimization, checkpointing, and FSDP unchanged.
    """

    _DEFAULT_LORA_TARGETS = (
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    )
    _PROFILE = DistributionMatchingProfile(
        supported_training_methods=frozenset({"dmd"}),
        supports_guidance=False,
        supports_ida=False,
        supports_diversity=False,
        supports_real_data_fake=False,
        supports_warped_denoising_schedule=False,
        default_latent_dtype=torch.float32,
    )

    def __init__(self, model, options: Mapping | None = None) -> None:
        super().__init__(model)
        options = MiniMaxH3DistributionMatchingOptions.from_mapping(options)
        self.video_weight = options.video_loss_weight
        self.audio_weight = options.audio_loss_weight
        self.video_shift = options.video_flow_shift
        self.audio_shift = options.audio_flow_shift
        self._layout_cache = {}

    @property
    def profile(self) -> DistributionMatchingProfile:
        return self._PROFILE

    @property
    def default_negative_prompt(self):
        return ""

    @property
    def default_lora_target_modules(self):
        return self._DEFAULT_LORA_TARGETS

    @property
    def generation_shape_dimensions(self) -> int:
        return 3

    def latent_shape(
        self,
        batch,
        generation_shapes,
        broadcast,
    ):
        prompt = batch["conditioning"].get("prompt", "")
        _require_single_prompt(prompt)
        num_frames, height, width = resolve_generation_shape(
            generation_shapes,
            batch,
            expected_dimensions=self.generation_shape_dimensions,
            broadcast=broadcast,
        )
        size_multiple = self.model.vae_spatial_scale_factor * self.model.patch_size[1]
        width_multiple = self.model.vae_spatial_scale_factor * self.model.patch_size[2]
        if height % size_multiple or width % width_multiple:
            raise ValueError(f"MiniMax-H3 height/width must be divisible by the VAE and patch scales ({size_multiple}, {width_multiple}), got {height}x{width}.")

        latent_frames = video_latent_num_frames(num_frames)
        latent_height = height // self.model.vae_spatial_scale_factor
        latent_width = width // self.model.vae_spatial_scale_factor
        patch_t, patch_h, patch_w = self.model.patch_size
        if patch_t != 1:
            raise ValueError(f"MiniMax-H3 T2AV requires temporal patch size 1, got {self.model.patch_size}.")
        video_rows = latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
        video_dimension = self.model.video_latent_channels * patch_t * patch_h * patch_w
        audio_latents = audio_latent_num_frames(num_frames)
        return MiniMaxH3LatentShape(
            num_frames=num_frames,
            latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            audio_latents=audio_latents,
            video_tokens=(1, video_rows, video_dimension),
            audio_tokens=(1, audio_latents * 2, self.model.audio_latent_channels),
        )

    def encode_conditions(
        self,
        batch,
        negative_prompt,
        guidance_scale,
        broadcast,
    ):
        del negative_prompt
        if guidance_scale != 1.0:
            raise ValueError("MiniMax-H3 is guidance-distilled; training.teacher.guidance_scale must be 1.0.")
        cached_condition = _cached_condition(batch, self.model)
        with torch.no_grad():
            condition = self.model.encode_condition(batch) if cached_condition is None else self.model.prepare_text_condition(cached_condition)
        return broadcast(condition), None

    def predict_velocity(self, latents, sigma, condition):
        self._validate_latents(latents)
        video_sigma, audio_sigma = self._modality_sigmas(sigma)
        layout = self._layout(condition, latents.shape)
        timesteps, timestep_indices = build_row_timesteps(
            layout,
            video_sigma,
            audio_sigma,
        )
        with self.model.transformer_forward_context():
            prediction = self.model.denoiser_module()(
                hidden_states=latents.video,
                audio_hidden_states=latents.audio,
                encoder_hidden_states=condition["prompt_embeds"],
                timestep=timesteps.to(self.device),
                timestep_indices=timestep_indices.to(self.device),
                token_tags=layout.token_tags,
                position_ids=layout.position_ids,
                video_indices=layout.video_indices,
                audio_indices=layout.audio_indices,
                text_indices=layout.text_indices,
                return_dict=False,
            )
        if not isinstance(prediction, (tuple, list)) or len(prediction) < 2:
            raise TypeError("MiniMax-H3 transformer must return (video_velocity, audio_velocity) when return_dict=False.")
        return MiniMaxH3JointLatents(
            video=prediction[0],
            audio=prediction[1],
            shape=latents.shape,
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
        del cfg_norm
        if negative_condition is not None or guidance_scale != 1.0:
            raise ValueError("MiniMax-H3 has no unconditional branch and only supports guidance_scale=1.0.")
        return self.predict_velocity(latents, sigma, condition)

    def initial_latents(self, latent_shape, dtype, broadcast):
        if latent_shape.video_tokens[0] != 1 or latent_shape.audio_tokens[0] != 1:
            raise ValueError(f"MiniMax-H3 DMD requires physical batch size 1, got {latent_shape}.")
        video = broadcast(torch.randn(latent_shape.video_tokens, device=self.device, dtype=dtype))
        audio = broadcast(torch.randn(latent_shape.audio_tokens, device=self.device, dtype=dtype))
        return MiniMaxH3JointLatents(video, audio, latent_shape)

    @staticmethod
    def latent_hw(latent_shape):
        del latent_shape
        return None

    @staticmethod
    def random_noise_like(latents, dtype, broadcast):
        return MiniMaxH3JointLatents(
            broadcast(torch.randn_like(latents.video, dtype=dtype)),
            broadcast(torch.randn_like(latents.audio, dtype=dtype)),
            latents.shape,
        )

    def add_noise(self, scheduler, latents, noise, sigma):
        del scheduler
        video_sigma, audio_sigma = self._modality_sigmas(sigma)
        return MiniMaxH3JointLatents(
            self._mix_noise(latents.video, noise.video, video_sigma),
            self._mix_noise(latents.audio, noise.audio, audio_sigma),
            latents.shape,
        )

    @staticmethod
    def training_target(latents, noise):
        # H3's time is t=1-sigma, so its velocity points from noise to x0.
        return MiniMaxH3JointLatents(
            latents.video.float() - noise.video.float(),
            latents.audio.float() - noise.audio.float(),
            latents.shape,
        )

    def step(self, scheduler, velocity, step_index, sample):
        sigma = scheduler.sigma_at(step_index, device=self.device, dtype=torch.float32)
        sigma_next = scheduler.sigma_at(int(step_index) + 1, device=self.device, dtype=torch.float32)
        video_sigma, audio_sigma = self._modality_sigmas(sigma)
        video_sigma_next, audio_sigma_next = self._modality_sigmas(sigma_next)
        return (
            MiniMaxH3JointLatents(
                self._cleanward_step(sample.video, velocity.video, video_sigma, video_sigma_next),
                self._cleanward_step(sample.audio, velocity.audio, audio_sigma, audio_sigma_next),
                sample.shape,
            ),
            MiniMaxH3JointLatents(
                self._cleanward_x0(sample.video, velocity.video, video_sigma),
                self._cleanward_x0(sample.audio, velocity.audio, audio_sigma),
                sample.shape,
            ),
        )

    def x0_from_velocity(self, sample, velocity, sigma):
        video_sigma, audio_sigma = self._modality_sigmas(sigma)
        return MiniMaxH3JointLatents(
            self._cleanward_x0(sample.video, velocity.video, video_sigma),
            self._cleanward_x0(sample.audio, velocity.audio, audio_sigma),
            sample.shape,
        )

    def regression_loss(self, prediction, target):
        video_loss = F.mse_loss(prediction.video.float(), target.video.float())
        audio_loss = F.mse_loss(prediction.audio.float(), target.audio.float())
        return self.video_weight * video_loss + self.audio_weight * audio_loss

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
        return MiniMaxH3JointLatents(
            value.video.detach(),
            value.audio.detach(),
            value.shape,
        )

    @staticmethod
    def to_dtype(value, dtype):
        return MiniMaxH3JointLatents(
            value.video.to(dtype=dtype),
            value.audio.to(dtype=dtype),
            value.shape,
        )

    def extract_real_latents(self, batch, dtype, broadcast):
        del batch, dtype, broadcast
        raise ValueError("MiniMax-H3 joint DMD does not support real-data fake loss.")

    def _modality_sigmas(self, sigma):
        sigma = torch.as_tensor(sigma, device=self.device, dtype=torch.float32)
        if sigma.ndim == 0:
            sigma = sigma.reshape(1)
        if sigma.numel() != 1:
            raise ValueError(f"MiniMax-H3 currently requires one shared base sigma, got shape {tuple(sigma.shape)}.")
        return _shift_sigma(sigma, self.video_shift), _shift_sigma(sigma, self.audio_shift)

    def _layout(self, condition, shape):
        tags = condition["text_token_tags"]
        if tags.ndim != 1:
            raise ValueError(f"MiniMax-H3 text_token_tags must be one-dimensional, got {tuple(tags.shape)}.")
        if not bool((tags == 1).all()):
            raise ValueError("MiniMax-H3 T2AV DMD currently supports text-only cached conditions.")
        key = (
            int(tags.numel()),
            shape.latent_frames,
            shape.latent_height,
            shape.latent_width,
            shape.audio_latents,
            self.model.patch_size,
            self.device,
        )
        layout = self._layout_cache.get(key)
        if layout is None:
            layout = build_packed_sequence(
                tags.detach().cpu(),
                shape.latent_frames,
                shape.latent_height,
                shape.latent_width,
                shape.audio_latents,
                self.model.patch_size,
            ).to(self.device)
            self._layout_cache[key] = layout
        return layout

    @staticmethod
    def _mix_noise(latent, noise, sigma):
        expanded = _expand_sigma(sigma, latent.ndim)
        return ((1.0 - expanded) * latent.float() + expanded * noise.float()).to(latent.dtype)

    @staticmethod
    def _cleanward_step(sample, velocity, sigma, sigma_next):
        current = _expand_sigma(sigma, sample.ndim)
        following = _expand_sigma(sigma_next, sample.ndim)
        return (sample.float() + (current - following) * velocity.float()).to(sample.dtype)

    @staticmethod
    def _cleanward_x0(sample, velocity, sigma):
        expanded = _expand_sigma(sigma, sample.ndim)
        return (sample.float() + expanded * velocity.float()).to(sample.dtype)

    @staticmethod
    def _validate_latents(latents):
        if not isinstance(latents, MiniMaxH3JointLatents):
            raise TypeError(f"Expected MiniMaxH3JointLatents, got {type(latents)!r}.")
        _require_singleton_tensor(latents.video, "MiniMax-H3 video latent")
        _require_singleton_tensor(latents.audio, "MiniMax-H3 audio latent")
