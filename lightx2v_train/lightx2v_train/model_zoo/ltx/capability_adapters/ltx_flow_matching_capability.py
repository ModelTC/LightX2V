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
from lightx2v_train.model_zoo.native.ltx2 import Modality

from .common import LTXLatentCodec


class LTXFlowMatchingCapability(BoundCapability, FlowMatchingSFTCapability):
    """Supervised flow matching for LTX's joint video/audio denoiser."""

    def __init__(self, model) -> None:
        super().__init__(model)
        config = model.config["training"].get("ltx2", {})
        self.video_loss_weight = float(config.get("video_loss_weight", 1.0))
        self.audio_loss_weight = float(config.get("audio_loss_weight", 1.0))
        self.default_fps = float(config.get("default_fps", 24.0))
        self.latent_codec = LTXLatentCodec(model, default_fps=self.default_fps)

    def compute_loss(
        self,
        batch: Mapping[str, Any],
        context: SFTStepContext,
    ) -> LossResult:
        inputs = batch["inputs"]
        conditions = batch["conditioning"].get("positive")
        if conditions is None:
            raise KeyError("LTX flow matching expects conditioning.positive.")
        video_data, _, video_latents, _, video_tokens, audio_tokens = self.latent_codec.prepare(
            inputs,
            context.running_dtype,
        )

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

        video_positions, audio_positions = self.latent_codec.positions(
            video_data,
            video_latents,
            audio_tokens,
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

    @staticmethod
    def _masked_mse(prediction, target, mask):
        target_length = target.shape[1]
        prediction = prediction[:, -target_length:, :]
        weights = mask[:, -target_length:].unsqueeze(-1).float()
        loss = (prediction.float() - target.float()).square() * weights
        return loss.mean(dim=(-2, -1)) / weights.mean(dim=(-2, -1)).clamp_min(1e-8)
