"""Flow-matching SFT for MiniMax-H3 joint video/audio latents."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lightx2v_train.model_capabilities import (
    BoundCapability,
    FlowMatchingSFTCapability,
    LossResult,
    SFTStepContext,
)
from lightx2v_train.model_zoo.capability_adapters.common import (
    _training_cache_data,
    _uses_prompt_dropout,
)
from lightx2v_train.model_zoo.native.minimax_h3 import (
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
)

from .common import MiniMaxH3JointLatents, MiniMaxH3LatentShape


@dataclass(frozen=True)
class MiniMaxH3FlowMatchingOptions:
    video_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0
    video_flow_shift: float = 6.0
    audio_flow_shift: float = 3.0

    @classmethod
    def from_mapping(cls, config: Mapping | None) -> "MiniMaxH3FlowMatchingOptions":
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise ValueError("model.capabilities.flow_matching must be a mapping.")
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


class MiniMaxH3FlowMatchingCapability(BoundCapability, FlowMatchingSFTCapability):
    """Expose H3's two-modality flow through the framework SFT contract."""

    def __init__(self, model, options: Mapping | None = None) -> None:
        super().__init__(model)
        options = MiniMaxH3FlowMatchingOptions.from_mapping(options)
        self.video_weight = options.video_loss_weight
        self.audio_weight = options.audio_loss_weight
        self.video_shift = options.video_flow_shift
        self.audio_shift = options.audio_flow_shift
        self._layout_cache = {}

    def encode_training_cache(self, batch):
        with torch.no_grad():
            latents = self._joint_latents(batch)
        prompt = batch["conditioning"]["prompt"]
        prompts = {"positive": prompt}
        if _uses_prompt_dropout(self.model):
            prompts["unconditional"] = self.model.unconditional_prompt
        return _training_cache_data(
            self.model,
            batch,
            inputs={
                "video_latents": {
                    "tokens": latents.video,
                    "latent_frames": latents.shape.latent_frames,
                    "latent_height": latents.shape.latent_height,
                    "latent_width": latents.shape.latent_width,
                },
                "audio_latents": latents.audio,
            },
            prompts=prompts,
        )

    def compute_loss(
        self,
        batch: Mapping,
        context: SFTStepContext,
    ) -> LossResult:
        scheduler = context.noise_scheduler
        if getattr(scheduler, "do_time_shift", False):
            raise ValueError("MiniMax-H3 SFT applies separate modality flow shifts; set scheduler.time_shift_settings.do_time_shift=false.")

        with torch.no_grad():
            latents = self._joint_latents(batch)
            latents = MiniMaxH3JointLatents(
                context.broadcast(latents.video),
                context.broadcast(latents.audio),
                latents.shape,
            )
            condition = context.broadcast(self._condition(batch))
            sigma = context.broadcast(scheduler.sample_timestep_or_sigma())
            sigma = torch.as_tensor(sigma, device=self.model.device, dtype=torch.float32).reshape(-1)
            if sigma.numel() != 1:
                raise ValueError(f"MiniMax-H3 SFT requires one shared base sigma, got shape {tuple(sigma.shape)}.")

            noise = MiniMaxH3JointLatents(
                context.broadcast(torch.randn_like(latents.video, dtype=self._latent_dtype)),
                context.broadcast(torch.randn_like(latents.audio, dtype=self._latent_dtype)),
                latents.shape,
            )
            video_sigma, audio_sigma = self._modality_sigmas(sigma)
            noisy = MiniMaxH3JointLatents(
                self._mix_noise(latents.video, noise.video, video_sigma),
                self._mix_noise(latents.audio, noise.audio, audio_sigma),
                latents.shape,
            )
            target = MiniMaxH3JointLatents(
                latents.video.float() - noise.video.float(),
                latents.audio.float() - noise.audio.float(),
                latents.shape,
            )

        prediction = self._predict(noisy, condition, video_sigma, audio_sigma)
        video_loss = F.mse_loss(prediction.video.float(), target.video)
        audio_loss = F.mse_loss(prediction.audio.float(), target.audio)
        loss = self.video_weight * video_loss + self.audio_weight * audio_loss
        return LossResult(
            loss=loss,
            metrics={
                "video_loss": video_loss.detach(),
                "audio_loss": audio_loss.detach(),
            },
        )

    @property
    def _latent_dtype(self):
        return getattr(self.model, "latent_dtype", torch.float32)

    def _condition(self, batch):
        conditioning = batch.get("conditioning", {})
        active = conditioning.get("active", "positive")
        cached = conditioning.get(active)
        if cached is None:
            condition = self.model.encode_condition(batch)
        else:
            condition = self.model.prepare_text_condition(cached)
        if not bool((condition["text_token_tags"] == 1).all()):
            raise ValueError("MiniMax-H3 T2AV SFT currently supports text-only cached conditions.")
        return condition

    def _joint_latents(self, batch):
        inputs = batch.get("inputs", {})
        video_value = inputs.get("video_latents")
        if video_value is None:
            video_value = inputs.get("latents")
        audio_value = inputs.get("audio_latents")
        if video_value is None and audio_value is None:
            encoded = self.model.encode_to_cache_latents(batch)
            video_value = encoded["video_latents"]
            audio_value = encoded["audio_latents"]
        elif video_value is None or audio_value is None:
            raise KeyError("MiniMax-H3 requires both inputs.video_latents and inputs.audio_latents.")

        video, latent_frames, latent_height, latent_width = self._video_tokens(video_value)
        audio = self._audio_tokens(audio_value)
        audio_latents = audio.shape[1] // 2
        num_frames = self._source_num_frames(latent_frames)
        expected_audio_latents = audio_latent_num_frames(num_frames)
        if audio_latents != expected_audio_latents:
            raise ValueError(f"MiniMax-H3 video geometry implies {expected_audio_latents} audio latents for {num_frames} source frames, but the cache contains {audio_latents}.")
        shape = MiniMaxH3LatentShape(
            num_frames=num_frames,
            latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            audio_latents=audio_latents,
            video_tokens=tuple(video.shape),
            audio_tokens=tuple(audio.shape),
        )
        return MiniMaxH3JointLatents(video.contiguous(), audio.contiguous(), shape)

    def _video_tokens(self, value):
        metadata = value if isinstance(value, Mapping) else {}
        tensor = self._cached_tensor(value, "video_latents")
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 3:
            latent_frames = self._metadata_int(metadata, "latent_frames")
            latent_height = self._metadata_int(metadata, "latent_height")
            latent_width = self._metadata_int(metadata, "latent_width")
            tokens = tensor.to(device=self.model.device, dtype=self._latent_dtype)
        elif tensor.ndim in {4, 5}:
            if tensor.ndim == 4:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[0] != 1 or tensor.shape[1] != self.model.video_latent_channels:
                raise ValueError(f"MiniMax-H3 raw video latents must have shape [1,{self.model.video_latent_channels},F,H,W], got {tuple(tensor.shape)}.")
            tensor = tensor.to(device=self.model.device, dtype=self._latent_dtype)
            latent_frames, latent_height, latent_width = map(int, tensor.shape[-3:])
            tokens = self._patchify_video(tensor)
        else:
            raise ValueError(f"MiniMax-H3 video latents must be patchified [B,N,D] or raw [B,C,F,H,W], got {tuple(tensor.shape)}.")

        expected_dimension = self.model.video_latent_channels * math.prod(self.model.patch_size)
        patch_t, patch_h, patch_w = self.model.patch_size
        if latent_frames % patch_t or latent_height % patch_h or latent_width % patch_w:
            raise ValueError(f"MiniMax-H3 video latent geometry {(latent_frames, latent_height, latent_width)} is not divisible by patch size {self.model.patch_size}.")
        expected_rows = (latent_frames // patch_t) * (latent_height // patch_h) * (latent_width // patch_w)
        if tokens.shape != (1, expected_rows, expected_dimension):
            raise ValueError(f"MiniMax-H3 video tokens must have shape {(1, expected_rows, expected_dimension)}, got {tuple(tokens.shape)}.")
        return tokens, latent_frames, latent_height, latent_width

    def _audio_tokens(self, value):
        tensor = self._cached_tensor(value, "audio_latents")
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[-1] != self.model.audio_latent_channels:
            raise ValueError(f"MiniMax-H3 audio latents must have shape [1,N,{self.model.audio_latent_channels}], got {tuple(tensor.shape)}.")
        if tensor.shape[1] % 2:
            raise ValueError(f"MiniMax-H3 stereo audio requires an even token count, got {tensor.shape[1]}.")
        return tensor.to(device=self.model.device, dtype=self._latent_dtype)

    def _predict(self, latents, condition, video_sigma, audio_sigma):
        layout = self._layout(condition, latents.shape)
        timesteps, timestep_indices = build_row_timesteps(layout, video_sigma, audio_sigma)
        with self.model.transformer_forward_context():
            output = self.model.denoiser_module()(
                hidden_states=latents.video,
                audio_hidden_states=latents.audio,
                encoder_hidden_states=condition["prompt_embeds"],
                timestep=timesteps.to(self.model.device),
                timestep_indices=timestep_indices.to(self.model.device),
                token_tags=layout.token_tags,
                position_ids=layout.position_ids,
                video_indices=layout.video_indices,
                audio_indices=layout.audio_indices,
                text_indices=layout.text_indices,
                return_dict=False,
            )
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise TypeError("MiniMax-H3 transformer must return (video_velocity, audio_velocity) when return_dict=False.")
        if output[0].shape != latents.video.shape or output[1].shape != latents.audio.shape:
            raise ValueError(
                "MiniMax-H3 transformer output shapes do not match the SFT latents: "
                f"video={tuple(output[0].shape)} vs {tuple(latents.video.shape)}, "
                f"audio={tuple(output[1].shape)} vs {tuple(latents.audio.shape)}."
            )
        return MiniMaxH3JointLatents(output[0], output[1], latents.shape)

    def _layout(self, condition, shape):
        tags = condition["text_token_tags"]
        key = (
            int(tags.numel()),
            shape.latent_frames,
            shape.latent_height,
            shape.latent_width,
            shape.audio_latents,
            self.model.patch_size,
            self.model.device,
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
            ).to(self.model.device)
            self._layout_cache[key] = layout
        return layout

    def _modality_sigmas(self, sigma):
        return self._shift_sigma(sigma, self.video_shift), self._shift_sigma(sigma, self.audio_shift)

    @staticmethod
    def _shift_sigma(sigma, shift):
        return shift * sigma / (1.0 + (shift - 1.0) * sigma)

    @staticmethod
    def _mix_noise(latent, noise, sigma):
        sigma = sigma.reshape(sigma.shape[0], *([1] * (latent.ndim - 1)))
        return ((1.0 - sigma) * latent.float() + sigma * noise.float()).to(latent.dtype)

    def _patchify_video(self, video):
        batch, channels, frames, height, width = video.shape
        patch_t, patch_h, patch_w = self.model.patch_size
        return (
            video.reshape(
                batch,
                channels,
                frames // patch_t,
                patch_t,
                height // patch_h,
                patch_h,
                width // patch_w,
                patch_w,
            )
            .permute(0, 2, 4, 6, 1, 3, 5, 7)
            .reshape(batch, -1, channels * patch_t * patch_h * patch_w)
        )

    @staticmethod
    def _cached_tensor(value, name):
        if isinstance(value, Mapping):
            value = value.get("tokens", value.get("latents"))
        if not torch.is_tensor(value):
            raise TypeError(f"MiniMax-H3 {name} cache must contain a tensor, got {type(value)!r}.")
        return value

    @staticmethod
    def _metadata_int(metadata, key):
        value = metadata.get(key)
        if value is None:
            raise KeyError(f"Patchified MiniMax-H3 video latents require {key} metadata.")
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"MiniMax-H3 {key} must be scalar, got shape {tuple(value.shape)}.")
            value = value.item()
        elif isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"MiniMax-H3 {key} must contain one value, got {len(value)}.")
            value = value[0]
        value = int(value)
        if value <= 0:
            raise ValueError(f"MiniMax-H3 {key} must be positive, got {value}.")
        return value

    @staticmethod
    def _source_num_frames(latent_frames):
        chunk_latents = latent_frames - 2
        if chunk_latents < 0 or chunk_latents % 5:
            raise ValueError(f"MiniMax-H3 latent frame count must be 5*n+2, got {latent_frames}.")
        return chunk_latents // 5 * 17 + 5
