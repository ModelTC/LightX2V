"""Distribution matching for MiniMax-H3 audio/video latents."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lightx2v_train.model_capabilities import DistributionMatchingProfile
from lightx2v_train.model_zoo.capability_adapters.common import (
    GenericDistributionMatchingCapability,
    _require_single_prompt,
    _require_singleton_tensor,
)
from lightx2v_train.model_zoo.native.minimax_h3 import (
    KEYFRAME_NOISE_AUG,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    keyframe_condition_noise,
    patchify_video_latents,
    video_latent_num_frames,
)
from lightx2v_train.trainers.dmd.math import pdmd_critic_loss, pdmd_loss_with_stats
from lightx2v_train.utils.generation_shapes import resolve_generation_shape

from .common import MiniMaxH3JointLatents, MiniMaxH3LatentShape


@dataclass(frozen=True)
class MiniMaxH3DistributionMatchingOptions:
    video_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0
    video_flow_shift: float = 6.0
    audio_flow_shift: float = 3.0
    audio_dmd_loss_weight: float | None = None

    @classmethod
    def from_mapping(cls, config: Mapping | None) -> "MiniMaxH3DistributionMatchingOptions":
        options = cls(**(config or {}))
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
        supported_training_methods=frozenset({"dmd", "pdmd", "dcm"}),
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
        self.audio_dmd_weight = self.audio_weight if options.audio_dmd_loss_weight is None else options.audio_dmd_loss_weight
        self.video_shift = options.video_flow_shift
        self.audio_shift = options.audio_flow_shift
        self.pdmd = model.config.get("training", {}).get("method") == "pdmd"
        self._layout_cache = OrderedDict()
        self._dmd_metrics = {}

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

    def validate_generation_shapes(self, generation_shapes):
        if generation_shapes is not None:
            super().validate_generation_shapes(generation_shapes)

    def latent_shape(
        self,
        batch,
        generation_shapes,
        broadcast,
    ):
        prompt = batch["conditioning"].get("prompt", "")
        _require_single_prompt(prompt)
        if generation_shapes is None:
            metadata = batch["meta"]
            num_frames, height, width = (int(broadcast(metadata[key])) for key in ("num_frames", "target_height", "target_width"))
        else:
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
            raise ValueError(f"MiniMax-H3 requires temporal patch size 1, got {self.model.patch_size}.")
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
        conditioning = batch["conditioning"]
        with torch.no_grad():
            role = conditioning.get("active", "positive")
            if role in conditioning:
                condition = self.model.prepare_text_condition({**conditioning.get("shared", {}), **conditioning[role]})
            else:
                condition = self.model.encode_condition(batch)
            if "condition_video_latents" in condition:
                condition = self._noise_visual_condition(condition)
        return broadcast(condition), None

    def _noise_visual_condition(self, condition):
        clean = condition["condition_video_latents"]
        sizes = condition.get("reference_sizes")
        if sizes is None:
            scale = self.model.vae_spatial_scale_factor
            sizes = [(condition["target_height"] // scale, condition["target_width"] // scale)] * len(condition["keyframe_anchors"])
        noise = keyframe_condition_noise(
            tuple((1, height, width) for height, width in sizes),
            self.model.patch_size,
            self.model.video_latent_channels,
            device=self.device,
        )
        condition["condition_video_latents"] = KEYFRAME_NOISE_AUG * clean + (1.0 - KEYFRAME_NOISE_AUG) * noise
        return condition

    def predict_velocity(self, latents, sigma, condition):
        video, audio = self._forward(latents, sigma, condition)
        condition_rows = condition.get("condition_video_latents")
        if condition_rows is not None:
            video = video[:, condition_rows.shape[0] :]
        return MiniMaxH3JointLatents(video, audio, latents.shape)

    def predict_velocity_at_sigmas(self, latents, sigma_pair, condition):
        if len(sigma_pair) != 2:
            raise ValueError("Expected one physical sigma per H3 modality.")
        return self.predict_velocity(latents, sigma_pair, condition)

    @property
    def gan_feature_dim(self):
        return self.model.denoiser_module().config.hidden_size

    def gan_geometry(self, latents):
        shape = latents.shape
        scale = self.model.vae_spatial_scale_factor
        return shape.num_frames, shape.latent_height * scale, shape.latent_width * scale

    def predict_gan_features(self, latents, sigma, condition, feature_block):
        return self._forward(latents, sigma, condition, feature_block=feature_block)

    def _forward(self, latents, sigma, condition, *, feature_block=None):
        self._validate_latents(latents)
        video_sigma, audio_sigma = self._modality_sigmas(sigma)
        layout = self._layout(condition, latents.shape)
        timesteps, timestep_indices = build_row_timesteps(
            layout,
            video_sigma,
            audio_sigma,
        )
        video = latents.video
        if layout.num_condition_video_rows:
            rows = condition["condition_video_latents"]
            if rows.shape[0] != layout.num_condition_video_rows:
                raise ValueError("H3 condition rows do not match the packed geometry.")
            video = torch.cat((rows.to(video).unsqueeze(0), video), dim=1)
        with self.model.transformer_forward_context():
            return self.model.denoiser_module()(
                hidden_states=video,
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
                feature_block=feature_block,
                target_video_indices=layout.video_indices[layout.num_condition_video_rows :],
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
        if self.pdmd:
            video_loss = pdmd_critic_loss(prediction.video, target.video)
            audio_loss = pdmd_critic_loss(prediction.audio, target.audio)
        else:
            video_loss = F.mse_loss(prediction.video.float(), target.video.float())
            audio_loss = F.mse_loss(prediction.audio.float(), target.audio.float())
        return self.video_weight * video_loss + self.audio_weight * audio_loss

    def dmd_loss(self, latents, fake_x0, teacher_x0):
        if self.pdmd:
            video, video_normalizer, video_rms, video_nonfinite = pdmd_loss_with_stats(latents.video, fake_x0.video, teacher_x0.video)
            audio, audio_normalizer, audio_rms, audio_nonfinite = pdmd_loss_with_stats(latents.audio, fake_x0.audio, teacher_x0.audio)
            video, audio = self.video_weight * video, self.audio_dmd_weight * audio
            self._dmd_metrics = {
                "dmd_video": video.detach(),
                "dmd_audio": audio.detach(),
                "dmd_video_normalizer": video_normalizer,
                "dmd_audio_normalizer": audio_normalizer,
                "dmd_video_direction_rms": video_rms,
                "dmd_audio_direction_rms": audio_rms,
                "dmd_nonfinite_update": video_nonfinite + audio_nonfinite,
            }
            return video + audio
        return self.video_weight * super().dmd_loss(
            latents.video,
            fake_x0.video,
            teacher_x0.video,
        ) + self.audio_dmd_weight * super().dmd_loss(
            latents.audio,
            fake_x0.audio,
            teacher_x0.audio,
        )

    def dmd_metrics(self):
        return self._dmd_metrics

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
        shape = self.latent_shape(batch, None, broadcast)
        latent_geometry = (shape.latent_frames, shape.latent_height, shape.latent_width)
        raw_video_shape = (self.model.video_latent_channels, *latent_geometry)
        raw_audio_shape = (2, self.model.audio_latent_channels, shape.audio_latents)
        inputs = batch.get("inputs", {})
        video = inputs.get("video_latents")
        if video is None:
            video = torch.load(batch["meta"]["video_latent_path"], map_location="cpu", weights_only=True)
        if isinstance(video, dict):
            if "tokens" in video:
                packed_geometry = (video["latent_frames"], video["latent_height"], video["latent_width"])
                if packed_geometry != latent_geometry:
                    raise ValueError(f"H3 cached video latent geometry {packed_geometry} does not match {latent_geometry}.")
                video = video["tokens"]
            else:
                if not video["normalized"]:
                    raise ValueError("H3 target video latents must be normalized.")
                video = video["latents"]
        audio = inputs.get("audio_latents")
        if audio is None:
            audio = torch.load(batch["meta"]["audio_latent_path"], map_location="cpu", weights_only=True)
        if isinstance(audio, dict):
            audio = audio["latents"]

        if video.shape == (1, *raw_video_shape):
            video = video[0]
        if video.shape == raw_video_shape:
            video = patchify_video_latents(video.unsqueeze(0), self.model.patch_size).unsqueeze(0)
        if audio.shape == (1, *raw_audio_shape):
            audio = audio[0]
        if audio.shape == raw_audio_shape:
            audio = audio.transpose(1, 2).reshape(shape.audio_tokens)
        if video.shape != shape.video_tokens or audio.shape != shape.audio_tokens:
            raise ValueError(f"H3 cached AV geometry mismatch: expected packed shapes {shape.video_tokens}/{shape.audio_tokens}, got {tuple(video.shape)}/{tuple(audio.shape)}.")

        return MiniMaxH3JointLatents(
            broadcast(video.to(self.device, dtype=dtype)),
            broadcast(audio.to(self.device, dtype=dtype)),
            shape,
        )

    def _modality_sigmas(self, sigma):
        if isinstance(sigma, (list, tuple)):
            sigma = torch.stack([torch.as_tensor(value, device=self.device, dtype=torch.float32).reshape(()) for value in sigma])
        sigma = torch.as_tensor(sigma, device=self.device, dtype=torch.float32).reshape(-1)
        if sigma.shape == (2,):
            return sigma[:1], sigma[1:]
        if sigma.numel() != 1:
            raise ValueError("H3 expects one base sigma or a pair of physical sigmas.")
        return _shift_sigma(sigma, self.video_shift), _shift_sigma(sigma, self.audio_shift)

    def _layout(self, condition, shape):
        tags = condition["text_token_tags"]
        anchors = condition.get("keyframe_anchors", ())
        references = condition.get("reference_sizes", ())
        scale = self.model.vae_spatial_scale_factor
        for key, expected in (("target_num_frames", shape.num_frames), ("target_height", shape.latent_height * scale), ("target_width", shape.latent_width * scale)):
            if key in condition and condition[key] != expected:
                raise ValueError(f"H3 cached {key}={condition[key]} does not match {expected}.")
        key = (
            tuple(tags.tolist()),
            anchors,
            references,
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
                keyframe_anchors=anchors,
                reference_sizes=references,
            ).to(self.device)
            self._layout_cache[key] = layout
            if len(self._layout_cache) > 16:
                self._layout_cache.popitem(last=False)
        else:
            self._layout_cache.move_to_end(key)
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
