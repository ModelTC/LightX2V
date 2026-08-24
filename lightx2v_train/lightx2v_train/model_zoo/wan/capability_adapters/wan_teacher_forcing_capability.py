"""Teacher-forcing capability for causal Wan models."""

from typing import Any, Mapping

import torch
import torch.distributed as dist

from lightx2v_train.model_capabilities import (
    BoundCapability,
    LossResult,
    TeacherForcingCapability,
    TeacherForcingStepContext,
)
from lightx2v_train.model_zoo.capability_adapters.common import _cached_condition
from lightx2v_train.model_zoo.wan.training_cache import encode_wan_video_cache
from lightx2v_train.runtime.distributed import (
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    is_sequence_parallel_enabled,
)
from lightx2v_train.runtime.sequence_parallel import (
    sequence_parallel_frame_slice,
)


class WanTeacherForcingCapability(BoundCapability, TeacherForcingCapability):
    """Chunk-wise teacher forcing for a causal Wan denoiser."""

    def encode_training_cache(self, batch):
        return encode_wan_video_cache(self.model, batch)

    def compute_loss(
        self,
        batch: Mapping[str, Any],
        context: TeacherForcingStepContext,
    ) -> LossResult:
        broadcast = context.broadcast

        with torch.no_grad():
            latent = self._latent(batch, context.running_dtype)
            _, _, num_frames, _, _ = latent.shape
            parallel_size = get_sequence_parallel_world_size() if is_sequence_parallel_enabled() else 1
            frame_multiple = parallel_size * context.num_frame_per_chunk
            if num_frames % frame_multiple:
                raise ValueError(
                    f"Teacher-forcing frames ({num_frames}) must be divisible by sequence_parallel_size * frames_per_chunk ({parallel_size} * {context.num_frame_per_chunk} = {frame_multiple})."
                )

            latent = broadcast(latent)
            noise = broadcast(torch.randn_like(latent, dtype=context.running_dtype))
            sigmas, weights = context.scheduler.sample_chunkwise(
                num_frames=num_frames,
                num_frame_per_chunk=context.num_frame_per_chunk,
                device=latent.device,
                dtype=context.running_dtype,
            )
            sigmas = broadcast(sigmas)
            weights = broadcast(weights)
            noisy_latent = context.scheduler.add_noise(latent, noise, sigmas)
            condition = _cached_condition(batch, self.model)
            if condition is None:
                condition = self.model.encode_condition(batch)
            condition = broadcast(condition)

            clean_latent = latent
            augmentation_sigmas = None
            if context.noise_augmentation_max_timestep > 0:
                augmentation_sigmas = context.scheduler.sample_clean_augmentation(
                    num_frames=num_frames,
                    num_frame_per_chunk=context.num_frame_per_chunk,
                    max_timestep=(context.noise_augmentation_max_timestep),
                    device=latent.device,
                    dtype=context.running_dtype,
                )
                augmentation_sigmas = broadcast(augmentation_sigmas)
                clean_latent = context.scheduler.add_noise(
                    latent,
                    noise,
                    augmentation_sigmas,
                )

            frame_start, frame_end, _ = sequence_parallel_frame_slice(
                num_frames,
                context.num_frame_per_chunk,
            )
            latent = latent[:, :, frame_start:frame_end].contiguous()
            noise = noise[:, :, frame_start:frame_end].contiguous()
            noisy_latent = noisy_latent[:, :, frame_start:frame_end].contiguous()
            clean_latent = clean_latent[:, :, frame_start:frame_end].contiguous()
            sigmas = sigmas[:, frame_start:frame_end].contiguous()
            weights = weights[:, frame_start:frame_end].contiguous()
            if augmentation_sigmas is not None:
                augmentation_sigmas = augmentation_sigmas[:, frame_start:frame_end].contiguous()

        prediction = self.model.denoise_teacher_forcing(
            noisy_latent,
            sigmas,
            condition,
            clean_latent=clean_latent,
            aug_timestep_or_sigma=augmentation_sigmas,
            frame_offset=frame_start,
            global_num_frames=num_frames,
        )
        target = noise - latent
        frame_loss = (prediction.float() - target.float()).square().mean(dim=(1, 3, 4))
        weighted_loss = (frame_loss * weights).sum()
        if is_sequence_parallel_enabled():
            dist.all_reduce(
                weighted_loss,
                op=dist.ReduceOp.SUM,
                group=get_sequence_parallel_group(),
            )
        return LossResult(loss=weighted_loss / num_frames)

    def _latent(self, batch, dtype):
        latent = batch["inputs"].get("latents")
        if latent is None:
            latent = self.model.encode_to_latent(batch)
        latent = latent.to(device=self.model.device, dtype=dtype)
        if latent.ndim == 4:
            latent = latent.unsqueeze(0)
        if latent.shape[0] != 1:
            raise ValueError("Wan teacher forcing only supports physical batch size 1.")

        channels = self.model._latent_channels()
        if latent.shape[1] != channels and latent.shape[2] == channels:
            latent = latent.permute(0, 2, 1, 3, 4).contiguous()
        if latent.shape[1] != channels:
            raise ValueError(f"Teacher-forcing latent channels ({latent.shape[1]}) do not match model channels ({channels}).")
        return latent
