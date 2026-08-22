"""Autoregressive distribution-matching capability for LTX models."""

from __future__ import annotations

import math

import torch

from lightx2v_train.model_capabilities import (
    AutoregressiveDistributionMatchingCapability,
    AutoregressiveRolloutContext,
    BoundCapability,
)
from lightx2v_train.model_zoo.native.ltx2 import Modality

from .common import LTXJointLatents
from .ltx_distribution_matching_capability import LTXDistributionMatchingCapability


class LTXAutoregressiveDistributionMatchingCapability(
    BoundCapability,
    AutoregressiveDistributionMatchingCapability,
):
    """Cached chunk-wise rollout for LTX joint video/audio states."""

    def __init__(self, model) -> None:
        super().__init__(model)
        self.distribution_matching = LTXDistributionMatchingCapability(model)

    def rollout(
        self,
        condition,
        latent_shape,
        initial_latents,
        context: AutoregressiveRolloutContext,
    ):
        latents = initial_latents
        if latents.video.shape[0] != 1 or latents.audio.shape[0] != 1:
            raise ValueError("LTX autoregressive DMD only supports physical batch size 1.")
        _, _, num_frames, latent_height, latent_width = latent_shape.video_latent
        frame_sequence_length = int(latent_height) * int(latent_width)
        if latents.video.shape[1] != num_frames * frame_sequence_length:
            if latents.video.shape[1] % num_frames:
                raise ValueError("LTX video token count is not divisible by latent frames.")
            frame_sequence_length = latents.video.shape[1] // num_frames
        if num_frames % context.frames_per_chunk:
            raise ValueError(f"LTX latent frames ({num_frames}) must be divisible by frames_per_chunk ({context.frames_per_chunk}).")

        transformer = self._unwrap_transformer()
        if transformer is None:
            raise RuntimeError("LTX autoregressive rollout requires transformer KV-cache inference support.")
        transformer.train()
        num_blocks = num_frames // context.frames_per_chunk
        exit_indices = self._sample_exit_indices(
            1 if context.same_step_across_blocks else num_blocks,
            len(context.denoising_steps),
            latents.video.device,
        )
        video_positions = self.distribution_matching._video_token_positions(latent_shape)
        audio_positions = self.distribution_matching._audio_token_positions(latent_shape)
        cache = self._new_caches(
            dtype=latents.video.dtype,
            device=latents.video.device,
            video_total_tokens=latents.video.shape[1],
            audio_total_tokens=latents.audio.shape[1],
            video_frame_sequence_length=frame_sequence_length,
            num_frames=num_frames,
            frames_per_chunk=context.frames_per_chunk,
        )

        video_chunks = []
        audio_chunks = []
        for block_index in range(num_blocks):
            frame_start = block_index * context.frames_per_chunk
            frame_end = frame_start + context.frames_per_chunk
            video_start = frame_start * frame_sequence_length
            video_end = frame_end * frame_sequence_length
            audio_start, audio_end = self._audio_range(
                block_index,
                num_blocks,
                latents.audio.shape[1],
            )
            block = LTXJointLatents(
                latents.video[:, video_start:video_end],
                latents.audio[:, audio_start:audio_end],
                latent_shape,
            )
            positions = (
                video_positions[:, :, video_start:video_end],
                audio_positions[:, :, audio_start:audio_end],
            )
            exit_index = int(exit_indices[0] if context.same_step_across_blocks else exit_indices[block_index])

            x0 = None
            for step_index in range(exit_index + 1):
                sigma = context.trajectory_scheduler.sigma_at(
                    step_index,
                    device=self.model.device,
                    dtype=context.running_dtype,
                )
                enable_grad = context.grad_enabled and step_index == exit_index
                grad_context = torch.enable_grad if enable_grad else torch.no_grad
                with grad_context():
                    velocity = self._predict_chunk(
                        block,
                        sigma,
                        condition,
                        positions,
                        cache,
                        video_start,
                        audio_start,
                    )
                    x0 = self.distribution_matching.x0_from_velocity(
                        block,
                        velocity,
                        sigma,
                    )
                if step_index < exit_index:
                    next_sigma = context.trajectory_scheduler.sigma_at(
                        step_index + 1,
                        device=self.model.device,
                        dtype=context.running_dtype,
                    )
                    with torch.no_grad():
                        noise = self.distribution_matching.random_noise_like(
                            x0,
                            x0.video.dtype,
                            lambda value: value,
                        )
                        block = self.distribution_matching.add_noise(
                            context.trajectory_scheduler,
                            x0,
                            noise,
                            next_sigma,
                        )

            video_chunks.append(x0.video)
            audio_chunks.append(x0.audio)
            cache_latents = self.distribution_matching.detach(x0)
            cache_sigma = torch.full(
                (1,),
                context.context_noise,
                device=self.model.device,
                dtype=context.running_dtype,
            )
            if context.context_noise > 0:
                cache_latents = self.distribution_matching.add_noise(
                    context.trajectory_scheduler,
                    cache_latents,
                    self.distribution_matching.random_noise_like(
                        cache_latents,
                        cache_latents.video.dtype,
                        lambda value: value,
                    ),
                    cache_sigma,
                )
            with torch.no_grad():
                self._predict_chunk(
                    cache_latents,
                    cache_sigma,
                    condition,
                    positions,
                    cache,
                    video_start,
                    audio_start,
                )

        generated = LTXJointLatents(
            torch.cat(video_chunks, dim=1).to(context.running_dtype),
            torch.cat(audio_chunks, dim=1).to(context.running_dtype),
            latent_shape,
        )
        return generated, exit_indices

    def _predict_chunk(
        self,
        latents,
        sigma,
        condition,
        positions,
        cache,
        video_start,
        audio_start,
    ):
        video_context, audio_context, context_mask = condition
        video_positions, audio_positions = positions
        if latents.video.shape[0] != 1 or latents.audio.shape[0] != 1:
            raise ValueError("LTX autoregressive DMD only supports physical batch size 1.")
        sigma = sigma.to(
            device=self.model.device,
            dtype=self.model.running_dtype,
        )
        video = Modality(
            enabled=True,
            latent=latents.video.to(
                device=self.model.device,
                dtype=self.model.running_dtype,
            ),
            sigma=sigma,
            timesteps=sigma.view(-1, 1)
            .expand(
                1,
                latents.video.shape[1],
            )
            .clone(),
            positions=video_positions.to(device=self.model.device),
            context=video_context.to(
                device=self.model.device,
                dtype=self.model.running_dtype,
            ),
            context_mask=context_mask.to(device=self.model.device),
        )
        audio = Modality(
            enabled=True,
            latent=latents.audio.to(
                device=self.model.device,
                dtype=self.model.running_dtype,
            ),
            sigma=sigma,
            timesteps=sigma.view(-1, 1)
            .expand(
                1,
                latents.audio.shape[1],
            )
            .clone(),
            positions=audio_positions.to(device=self.model.device),
            context=audio_context.to(
                device=self.model.device,
                dtype=self.model.running_dtype,
            ),
            context_mask=context_mask.to(device=self.model.device),
        )
        with self.model.transformer_forward_context():
            video_prediction, audio_prediction = self.model.denoiser_module()(
                video=video,
                audio=audio,
                perturbations=None,
                kv_cache=cache,
                video_current_start=video_start,
                audio_current_start=audio_start,
            )
        return LTXJointLatents(
            video_prediction,
            audio_prediction,
            latents.shape,
        )

    def _new_caches(
        self,
        dtype,
        device,
        video_total_tokens,
        audio_total_tokens,
        video_frame_sequence_length,
        num_frames,
        frames_per_chunk,
    ):
        transformer = self._unwrap_transformer()
        num_layers = len(transformer.transformer_blocks)
        video_heads = int(transformer.num_attention_heads)
        video_head_dim = int(transformer.inner_dim / video_heads)
        audio_heads = int(transformer.audio_num_attention_heads)
        audio_head_dim = int(transformer.audio_inner_dim / audio_heads)
        local_attention_size = int(getattr(transformer, "local_attn_size", -1))
        sink_size = int(getattr(transformer, "sink_size", 0))
        video_chunk_tokens = frames_per_chunk * video_frame_sequence_length
        if local_attention_size == -1:
            video_cache_size = video_total_tokens
            video_window = video_total_tokens
        else:
            video_window = max(
                video_chunk_tokens,
                local_attention_size * video_frame_sequence_length,
            )
            video_cache_size = max(
                video_window,
                sink_size * video_frame_sequence_length + video_chunk_tokens,
            )
        audio_tokens_per_frame = max(
            1,
            math.ceil(audio_total_tokens / max(1, num_frames)),
        )
        audio_chunk_tokens = max(
            1,
            math.ceil(audio_total_tokens / (num_frames // frames_per_chunk)),
        )
        if local_attention_size == -1:
            audio_cache_size = audio_total_tokens
            audio_window = audio_total_tokens
        else:
            audio_window = max(
                audio_chunk_tokens,
                local_attention_size * audio_tokens_per_frame,
            )
            audio_cache_size = max(
                audio_window,
                sink_size * audio_tokens_per_frame + audio_chunk_tokens,
            )
        return {
            "video": [
                self._empty_cache(
                    video_cache_size,
                    video_heads,
                    video_head_dim,
                    dtype,
                    device,
                    video_window,
                    sink_size * video_frame_sequence_length,
                )
                for _ in range(num_layers)
            ],
            "audio": [
                self._empty_cache(
                    audio_cache_size,
                    audio_heads,
                    audio_head_dim,
                    dtype,
                    device,
                    audio_window,
                    sink_size * audio_tokens_per_frame,
                )
                for _ in range(num_layers)
            ],
        }

    @staticmethod
    def _empty_cache(
        cache_size,
        heads,
        head_dim,
        dtype,
        device,
        attention_window_size,
        sink_tokens,
    ):
        return {
            "k": torch.zeros(
                (1, cache_size, heads, head_dim),
                dtype=dtype,
                device=device,
            ),
            "v": torch.zeros(
                (1, cache_size, heads, head_dim),
                dtype=dtype,
                device=device,
            ),
            "global_end_index": torch.zeros(
                1,
                dtype=torch.long,
                device=device,
            ),
            "local_end_index": torch.zeros(
                1,
                dtype=torch.long,
                device=device,
            ),
            "attention_window_size": int(attention_window_size),
            "sink_tokens": int(sink_tokens),
        }

    @staticmethod
    def _audio_range(block_index, num_blocks, total_tokens):
        start = (block_index * total_tokens + num_blocks - 1) // num_blocks
        end = ((block_index + 1) * total_tokens + num_blocks - 1) // num_blocks
        return start, end

    @staticmethod
    def _sample_exit_indices(count, num_steps, device):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                indices = torch.randint(0, num_steps, (count,), device=device)
            else:
                indices = torch.empty(count, dtype=torch.long, device=device)
            torch.distributed.broadcast(indices, src=0)
            return indices.tolist()
        return torch.randint(0, num_steps, (count,), device=device).tolist()

    def _unwrap_transformer(self):
        transformer = self.model.denoiser_module()
        for candidate in (
            transformer,
            getattr(transformer, "module", None),
            getattr(transformer, "_fsdp_wrapped_module", None),
        ):
            if candidate is not None and hasattr(candidate, "_forward_inference"):
                return candidate
        return None
