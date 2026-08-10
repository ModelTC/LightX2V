"""Flow-matching capability for LTX joint video/audio models."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from lightx2v_train.model_capabilities import (
    BoundCapability,
    FlowMatchingSFTCapability,
    LossResult,
    SFTStepContext,
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


class LTXFlowMatchingCapability(BoundCapability, FlowMatchingSFTCapability):
    """Supervised flow matching for LTX's joint video/audio denoiser."""

    def __init__(self, model) -> None:
        super().__init__(model)
        config = model.config["training"].get("ltx2", {})
        self.video_loss_weight = float(config.get("video_loss_weight", 1.0))
        self.audio_loss_weight = float(config.get("audio_loss_weight", 1.0))
        self.default_fps = float(config.get("default_fps", 24.0))
        self.video_patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)

    def compute_loss(
        self,
        batch: Mapping[str, Any],
        context: SFTStepContext,
    ) -> LossResult:
        inputs = batch["inputs"]
        video_data = inputs.get("video_latents")
        audio_data = inputs.get("audio_latents")
        conditions = batch["conditioning"].get("positive")
        if video_data is None or audio_data is None or conditions is None:
            raise KeyError("LTX flow matching expects inputs.video_latents, inputs.audio_latents, and conditioning.positive.")

        video_latents = self._video_latents(video_data, context.running_dtype)
        audio_latents = self._audio_latents(audio_data, context.running_dtype)
        video_tokens = self.video_patchifier.patchify(video_latents)
        audio_tokens = self.audio_patchifier.patchify(audio_latents)

        video_context, audio_context, context_mask = self.model.prepare_text_condition(conditions)
        video_noisy = self._initialize_noisy_target(
            video_tokens,
            context.noise_scheduler,
        )
        audio_noisy = self._initialize_noisy_target(
            audio_tokens,
            context.noise_scheduler,
            sigmas=video_noisy[-1],
        )
        video_input, video_target, video_timesteps, video_mask, video_sigmas = video_noisy
        audio_input, audio_target, audio_timesteps, audio_mask, audio_sigmas = audio_noisy

        video_positions = self._video_positions(
            video_data,
            video_latents,
        )
        audio_positions = self.audio_patchifier.get_patch_grid_bounds(
            output_shape=AudioLatentShape(
                frames=audio_tokens.shape[1],
                mel_bins=16,
                batch=1,
                channels=8,
            ),
            device=audio_tokens.device,
        )
        video = Modality(
            enabled=True,
            latent=video_input,
            sigma=video_sigmas,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_context,
            context_mask=context_mask,
        )
        audio = Modality(
            enabled=True,
            latent=audio_input,
            sigma=audio_sigmas,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_context,
            context_mask=context_mask,
        )

        with self.model.transformer_forward_context():
            video_prediction, audio_prediction = self.model.denoiser_module()(
                video=video,
                audio=audio,
                perturbations=None,
            )

        video_loss = self._masked_mse(
            video_prediction,
            video_target,
            video_mask,
        ).mean()
        audio_loss = self._masked_mse(
            audio_prediction,
            audio_target,
            audio_mask,
        ).mean()
        loss = self.video_loss_weight * video_loss + self.audio_loss_weight * audio_loss
        return LossResult(
            loss=loss,
            metrics={
                "video_loss": video_loss.detach(),
                "audio_loss": audio_loss.detach(),
            },
        )

    def _video_latents(self, data, dtype):
        latents = data["latents"].to(device=self.model.device, dtype=dtype)
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)
        if latents.ndim != 5:
            raise ValueError(f"LTX video latents must have shape [B,C,F,H,W] or [C,F,H,W], got {tuple(latents.shape)}.")
        if latents.shape[0] != 1:
            raise ValueError("LTX flow matching only supports physical batch size 1.")
        return latents.contiguous()

    def _audio_latents(self, data, dtype):
        latents = data["latents"].to(device=self.model.device, dtype=dtype)
        if latents.ndim == 3:
            latents = latents.unsqueeze(0)
        if latents.ndim != 4:
            raise ValueError(f"LTX audio latents must have shape [B,C,T,F] or [C,T,F], got {tuple(latents.shape)}.")
        if latents.shape[0] != 1:
            raise ValueError("LTX flow matching only supports physical batch size 1.")
        return latents.contiguous()

    @staticmethod
    def _initialize_noisy_target(latents: Tensor, scheduler, sigmas=None):
        _, sequence_length, _ = latents.shape
        if sigmas is None:
            sigmas = scheduler.sample_timestep_or_sigma(
                seq_len=sequence_length,
            )
        sigmas = sigmas.to(device=latents.device, dtype=latents.dtype)
        noise = torch.randn_like(latents)
        expanded = sigmas.view(-1, 1, 1)
        noisy = (1 - expanded) * latents + expanded * noise
        target = noise - latents
        timesteps = (
            sigmas.view(-1, 1)
            .expand(
                1,
                sequence_length,
            )
            .clone()
        )
        mask = torch.ones(
            1,
            sequence_length,
            dtype=torch.bool,
            device=latents.device,
        )
        return noisy, target, timesteps, mask, sigmas

    def _video_positions(self, data, latents):
        frames = int(data["num_frames"][0].item()) if "num_frames" in data else latents.shape[2]
        height = int(data["height"][0].item()) if "height" in data else latents.shape[3]
        width = int(data["width"][0].item()) if "width" in data else latents.shape[4]
        fps = float(data["fps"][0].item()) if "fps" in data else self.default_fps
        latent_coords = self.video_patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                frames=frames,
                height=height,
                width=width,
                batch=1,
                channels=128,
            ),
            device=latents.device,
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=SpatioTemporalScaleFactors.default(),
            causal_fix=True,
        ).float()
        positions[:, 0, ...] /= fps
        return positions

    @staticmethod
    def _masked_mse(prediction, target, mask):
        target_length = target.shape[1]
        prediction = prediction[:, -target_length:, :]
        weights = mask[:, -target_length:].unsqueeze(-1).float()
        loss = (prediction.float() - target.float()).square() * weights
        return loss.mean(dim=(-2, -1)) / weights.mean(dim=(-2, -1)).clamp_min(1e-8)
