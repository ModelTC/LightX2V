"""Teacher-forcing capability for LTX models."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from lightx2v_train.model_capabilities import (
    LossResult,
    TeacherForcingCapability,
    TeacherForcingStepContext,
)
from lightx2v_train.model_zoo.native.ltx2 import AudioLatentShape, Modality
from lightx2v_train.runtime.distributed import is_sequence_parallel_enabled

from .ltx_flow_matching_capability import LTXFlowMatchingCapability


class LTXTeacherForcingCapability(
    LTXFlowMatchingCapability,
    TeacherForcingCapability,
):
    """Chunk-wise teacher forcing for LTX's joint video/audio denoiser."""

    def __init__(self, model) -> None:
        super().__init__(model)
        self._mask_cache = {}

    def compute_loss(
        self,
        batch: Mapping[str, Any],
        context: TeacherForcingStepContext,
    ) -> LossResult:
        if is_sequence_parallel_enabled():
            raise ValueError("LTX teacher forcing does not support sequence parallel.")

        inputs = batch["inputs"]
        video_data = inputs.get("video_latents")
        audio_data = inputs.get("audio_latents")
        conditions = batch["conditioning"].get("positive")
        if video_data is None or audio_data is None or conditions is None:
            raise KeyError("LTX teacher forcing expects inputs.video_latents, inputs.audio_latents, and conditioning.positive.")

        with torch.no_grad():
            video_latents = self._video_latents(
                video_data,
                context.running_dtype,
            )
            audio_latents = self._audio_latents(
                audio_data,
                context.running_dtype,
            )
            video_latents, audio_latents = self._crop_to_chunk_multiple(
                video_latents,
                audio_latents,
                context.num_frame_per_chunk,
            )
            num_video_frames = int(video_latents.shape[2])
            video_tokens = self.video_patchifier.patchify(video_latents)
            audio_tokens = self.audio_patchifier.patchify(audio_latents)
            video_context, audio_context, context_mask = self.model.prepare_text_condition(conditions)

            video_inputs = self._prepare_video_inputs(
                video_tokens,
                num_frames=num_video_frames,
                spatial_tokens=int(video_latents.shape[3] * video_latents.shape[4]),
                context=context,
            )
            (
                video_input,
                video_target,
                video_timesteps,
                video_weights,
                video_sigmas,
                video_frame_weights,
                video_augmentation_sigmas,
                video_attention_mask,
                video_block_ids,
            ) = video_inputs
            audio_inputs = self._prepare_audio_inputs(
                audio_tokens,
                frame_sigmas=video_sigmas,
                frame_weights=video_frame_weights,
                augmentation_frame_sigmas=video_augmentation_sigmas,
                num_video_frames=num_video_frames,
                frames_per_chunk=context.num_frame_per_chunk,
            )
            (
                audio_input,
                audio_target,
                audio_timesteps,
                audio_weights,
                audio_sigmas,
                audio_attention_mask,
                audio_block_ids,
            ) = audio_inputs

            video_positions = self._video_positions(
                video_data,
                video_latents,
            )
            video_positions = torch.cat(
                [video_positions, video_positions],
                dim=2,
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
            audio_positions = torch.cat(
                [audio_positions, audio_positions],
                dim=2,
            )
            video_cross_mask = self._cross_attention_mask(
                video_tokens.device,
                video_block_ids,
                audio_block_ids,
            )
            audio_cross_mask = self._cross_attention_mask(
                audio_tokens.device,
                audio_block_ids,
                video_block_ids,
            )

        video = Modality(
            enabled=True,
            latent=video_input,
            sigma=video_sigmas[:, 0],
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_context,
            context_mask=context_mask,
            attention_mask=video_attention_mask,
            cross_attention_mask=video_cross_mask,
        )
        audio = Modality(
            enabled=True,
            latent=audio_input,
            sigma=audio_sigmas[:, 0],
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_context,
            context_mask=context_mask,
            attention_mask=audio_attention_mask,
            cross_attention_mask=audio_cross_mask,
        )
        with self.model.transformer_forward_context():
            video_prediction, audio_prediction = self.model.denoiser_module()(
                video=video,
                audio=audio,
                perturbations=None,
            )

        video_loss = self._weighted_token_mse(
            video_prediction,
            video_target,
            video_weights,
        ).mean()
        audio_loss = self._weighted_token_mse(
            audio_prediction,
            audio_target,
            audio_weights,
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
    def _crop_to_chunk_multiple(video, audio, frames_per_chunk):
        if frames_per_chunk <= 1:
            return video, audio
        num_frames = int(video.shape[2])
        keep_frames = num_frames // frames_per_chunk * frames_per_chunk
        if keep_frames == num_frames:
            return video, audio
        if keep_frames <= 0:
            raise ValueError(f"LTX sample has {num_frames} latent frames, fewer than frames_per_chunk={frames_per_chunk}.")
        audio_frames = int(audio.shape[2])
        keep_audio_frames = max(1, audio_frames * keep_frames // num_frames)
        return (
            video[:, :, :keep_frames].contiguous(),
            audio[:, :, :keep_audio_frames].contiguous(),
        )

    def _prepare_video_inputs(
        self,
        tokens,
        num_frames,
        spatial_tokens,
        context,
    ):
        frame_sigmas, frame_weights = context.scheduler.sample_chunkwise(
            num_frames=num_frames,
            num_frame_per_chunk=context.num_frame_per_chunk,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        token_sigmas = frame_sigmas.repeat_interleave(spatial_tokens, dim=1)
        token_weights = frame_weights.repeat_interleave(spatial_tokens, dim=1)
        noisy, target, noise = self._add_token_noise(tokens, token_sigmas)
        clean = tokens
        clean_timesteps = torch.zeros_like(token_sigmas)
        augmentation_frame_sigmas = None
        if context.noise_augmentation_max_timestep > 0:
            augmentation_frame_sigmas = context.scheduler.sample_clean_augmentation(
                num_frames=num_frames,
                num_frame_per_chunk=context.num_frame_per_chunk,
                max_timestep=context.noise_augmentation_max_timestep,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            clean_timesteps = augmentation_frame_sigmas.repeat_interleave(
                spatial_tokens,
                dim=1,
            )
            clean = (1.0 - clean_timesteps.unsqueeze(-1)) * tokens + clean_timesteps.unsqueeze(-1) * noise
        frame_ids = torch.arange(
            num_frames,
            device=tokens.device,
        ).repeat_interleave(spatial_tokens)
        block_ids = frame_ids // context.num_frame_per_chunk
        attention_mask = self._attention_mask(tokens.device, block_ids)
        return (
            torch.cat([clean, noisy], dim=1),
            target,
            torch.cat([clean_timesteps, token_sigmas], dim=1),
            token_weights,
            frame_sigmas,
            frame_weights,
            augmentation_frame_sigmas,
            attention_mask,
            block_ids,
        )

    def _prepare_audio_inputs(
        self,
        tokens,
        frame_sigmas,
        frame_weights,
        augmentation_frame_sigmas,
        num_video_frames,
        frames_per_chunk,
    ):
        _, sequence_length, _ = tokens.shape
        frame_ids = (torch.arange(sequence_length, device=tokens.device) * max(1, num_video_frames) // max(1, sequence_length)).clamp(max=frame_sigmas.shape[1] - 1)
        gather_ids = frame_ids.unsqueeze(0)
        token_sigmas = frame_sigmas.gather(1, gather_ids)
        token_weights = frame_weights.gather(1, gather_ids)
        noisy, target, noise = self._add_token_noise(tokens, token_sigmas)
        clean = tokens
        clean_timesteps = torch.zeros_like(token_sigmas)
        if augmentation_frame_sigmas is not None:
            clean_timesteps = augmentation_frame_sigmas.gather(1, gather_ids)
            clean = (1.0 - clean_timesteps.unsqueeze(-1)) * tokens + clean_timesteps.unsqueeze(-1) * noise
        block_ids = frame_ids // frames_per_chunk
        attention_mask = self._attention_mask(tokens.device, block_ids)
        return (
            torch.cat([clean, noisy], dim=1),
            target,
            torch.cat([clean_timesteps, token_sigmas], dim=1),
            token_weights,
            token_sigmas,
            attention_mask,
            block_ids,
        )

    def _attention_mask(self, device, block_ids):
        key = ("self", self._device_key(device), self._block_key(block_ids))
        if key not in self._mask_cache:
            self._mask_cache[key] = self._call_mask_builder(
                "prepare_teacher_forcing_mask_from_block_ids",
                device,
                block_ids,
            )
        return self._mask_cache[key]

    def _cross_attention_mask(self, device, query_ids, key_ids):
        key = (
            "cross",
            self._device_key(device),
            self._block_key(query_ids),
            self._block_key(key_ids),
        )
        if key not in self._mask_cache:
            self._mask_cache[key] = self._call_mask_builder(
                "prepare_teacher_forcing_cross_mask_from_block_ids",
                device,
                query_ids,
                key_ids,
            )
        return self._mask_cache[key]

    def _call_mask_builder(self, method_name, *args):
        transformer = self.model.denoiser_module()
        candidates = (
            transformer,
            getattr(transformer, "module", None),
            getattr(transformer, "_fsdp_wrapped_module", None),
        )
        for candidate in candidates:
            method = getattr(candidate, method_name, None)
            if method is not None:
                return method(*args)
        raise RuntimeError(f"LTX transformer does not implement {method_name}().")

    @staticmethod
    def _device_key(device):
        device = torch.device(device)
        index = device.index
        if device.type == "cuda" and index is None and torch.cuda.is_available():
            index = torch.cuda.current_device()
        return device.type, index

    @staticmethod
    def _block_key(block_ids):
        values = block_ids.detach().to(dtype=torch.long)
        if values.numel() == 0:
            return 0, None, None, None, None, None
        positions = torch.arange(
            values.numel(),
            device=values.device,
            dtype=torch.long,
        )
        return (
            int(values.numel()),
            int(values[0].item()),
            int(values[-1].item()),
            int(values.max().item()),
            int(values.sum().item()),
            int((values * positions).sum().item()),
        )

    @staticmethod
    def _add_token_noise(tokens, sigmas, noise=None):
        if noise is None:
            noise = torch.randn_like(tokens)
        expanded = sigmas.unsqueeze(-1)
        noisy = (1.0 - expanded) * tokens + expanded * noise
        return noisy, noise - tokens, noise

    @staticmethod
    def _weighted_token_mse(prediction, target, weights):
        prediction = prediction[:, -target.shape[1] :, :]
        loss = (prediction.float() - target.float()).square().mean(dim=-1)
        weights = weights.float()
        return (loss * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-8)
