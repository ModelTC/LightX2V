"""Autoregressive distribution-matching capability for causal Wan models."""

from __future__ import annotations

import torch
import torch.distributed as dist

from lightx2v_train.model_capabilities import (
    AutoregressiveDistributionMatchingCapability,
    AutoregressiveRolloutContext,
    BoundCapability,
)
from lightx2v_train.runtime.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    is_sequence_parallel_enabled,
)
from lightx2v_train.runtime.sequence_parallel import all_gather_sequence


class WanAutoregressiveDistributionMatchingCapability(
    BoundCapability,
    AutoregressiveDistributionMatchingCapability,
):
    """Cached chunk-wise rollout for causal Wan transformers."""

    def rollout(
        self,
        condition,
        latent_shape,
        initial_latents,
        context: AutoregressiveRolloutContext,
    ):
        latents = initial_latents
        _, _, num_frames, _, _ = latents.shape
        if latents.shape[0] != 1:
            raise ValueError("Wan autoregressive DMD only supports physical batch size 1.")
        if num_frames % context.frames_per_chunk:
            raise ValueError(f"Autoregressive latent frames ({num_frames}) must be divisible by frames_per_chunk ({context.frames_per_chunk}).")

        use_sp_cache = bool(context.sequence_parallel_cache and is_sequence_parallel_enabled())
        parallel_size = get_sequence_parallel_world_size() if use_sp_cache else 1
        parallel_rank = get_sequence_parallel_rank() if use_sp_cache else 0
        if use_sp_cache and context.frames_per_chunk % parallel_size:
            raise ValueError("frames_per_chunk must be divisible by sequence parallel size when sp_cache is enabled.")
        transformer = self.model.denoiser_module()
        if use_sp_cache and bool(getattr(transformer, "defer_kv_cache_updates", False)):
            raise ValueError("sp_cache is incompatible with deferred KV-cache updates.")

        transformer.train()
        model_context = self.model._condition_to_context_tensor(
            condition,
            batch_size=1,
        )
        frame_sequence_length = self._frame_sequence_length(latents)
        kv_cache, cross_attention_cache = self._new_caches(
            latents.dtype,
            latents.device,
            num_frames,
            frame_sequence_length,
            sequence_parallel_cache=use_sp_cache,
        )
        num_blocks = num_frames // context.frames_per_chunk
        exit_indices = self._sample_exit_indices(
            1 if context.same_step_across_blocks else num_blocks,
            len(context.denoising_steps),
            latents.device,
        )

        output_chunks = []
        current_start_frame = 0
        for block_index in range(num_blocks):
            if use_sp_cache:
                local_frames = context.frames_per_chunk // parallel_size
                local_start = current_start_frame + parallel_rank * local_frames
            else:
                local_frames = context.frames_per_chunk
                local_start = current_start_frame
            block_latents = latents[:, :, local_start : local_start + local_frames]
            exit_index = int(exit_indices[0] if context.same_step_across_blocks else exit_indices[block_index])

            x0 = None
            for step_index, current_timestep in enumerate(context.denoising_steps):
                timestep = torch.full(
                    (1, local_frames),
                    float(current_timestep),
                    device=latents.device,
                    dtype=torch.float32,
                )
                enable_grad = context.grad_enabled and step_index == exit_index
                grad_context = torch.enable_grad if enable_grad else torch.no_grad
                with grad_context():
                    flow = self._forward_chunk(
                        block_latents,
                        timestep,
                        model_context,
                        kv_cache,
                        cross_attention_cache,
                        current_start=current_start_frame * frame_sequence_length,
                        cache_start=current_start_frame * frame_sequence_length,
                        local_frame_offset=local_start,
                        balanced_sequence_parallel=use_sp_cache,
                    )
                    x0 = self._flow_to_x0(
                        block_latents,
                        flow,
                        timestep,
                        context.denoising_scheduler,
                    )
                if step_index == exit_index:
                    break
                next_timestep = torch.full(
                    (1, local_frames),
                    float(context.denoising_steps[step_index + 1]),
                    device=latents.device,
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    block_latents = self._add_noise(
                        x0,
                        torch.randn_like(x0),
                        next_timestep,
                        context.denoising_scheduler,
                    )

            output_chunks.append(all_gather_sequence(x0, dim=2) if use_sp_cache else x0)
            cache_latents = x0.detach()
            cache_timestep = torch.full(
                (1, local_frames),
                context.context_noise,
                device=latents.device,
                dtype=torch.float32,
            )
            if context.context_noise > 0:
                cache_latents = self._add_noise(
                    cache_latents,
                    torch.randn_like(cache_latents),
                    cache_timestep,
                    context.denoising_scheduler,
                )
            with torch.no_grad():
                self._forward_chunk(
                    cache_latents,
                    cache_timestep,
                    model_context,
                    kv_cache,
                    cross_attention_cache,
                    current_start=current_start_frame * frame_sequence_length,
                    cache_start=current_start_frame * frame_sequence_length,
                    local_frame_offset=local_start,
                    balanced_sequence_parallel=use_sp_cache,
                )
            current_start_frame += context.frames_per_chunk

        generated = torch.cat(output_chunks, dim=2).to(dtype=context.running_dtype)
        return generated, exit_indices

    def _forward_chunk(
        self,
        latents,
        timestep,
        condition,
        kv_cache,
        cross_attention_cache,
        **kwargs,
    ):
        with self.model.transformer_forward_context():
            return self.model.denoiser_module()(
                latents,
                t=timestep,
                context=condition,
                seq_len=self.model._sequence_length(latents),
                kv_cache=kv_cache,
                crossattn_cache=cross_attention_cache,
                **kwargs,
            )

    def _new_caches(
        self,
        dtype,
        device,
        num_frames,
        frame_sequence_length,
        sequence_parallel_cache,
    ):
        transformer = self.model.denoiser_module()
        num_layers = int(getattr(transformer, "num_layers", len(transformer.blocks)))
        num_heads = int(transformer.num_heads)
        head_dim = int(transformer.dim // transformer.num_heads)
        kv_heads = num_heads
        if sequence_parallel_cache:
            parallel_size = get_sequence_parallel_world_size()
            if num_heads % parallel_size:
                raise ValueError(f"Transformer heads ({num_heads}) must be divisible by sequence parallel size ({parallel_size}).")
            kv_heads = num_heads // parallel_size
        local_attention_size = int(getattr(transformer, "local_attn_size", -1))
        cache_size = num_frames * frame_sequence_length if local_attention_size == -1 else local_attention_size * frame_sequence_length
        kv_cache = [
            {
                "k": torch.zeros(
                    (1, cache_size, kv_heads, head_dim),
                    dtype=dtype,
                    device=device,
                ),
                "v": torch.zeros(
                    (1, cache_size, kv_heads, head_dim),
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
            }
            for _ in range(num_layers)
        ]
        cross_attention_cache = [
            {
                "k": torch.zeros(
                    (
                        1,
                        self.model.max_sequence_length,
                        num_heads,
                        head_dim,
                    ),
                    dtype=dtype,
                    device=device,
                ),
                "v": torch.zeros(
                    (
                        1,
                        self.model.max_sequence_length,
                        num_heads,
                        head_dim,
                    ),
                    dtype=dtype,
                    device=device,
                ),
                "is_init": False,
            }
            for _ in range(num_layers)
        ]
        return kv_cache, cross_attention_cache

    def _frame_sequence_length(self, latent):
        _, _, _, height, width = latent.shape
        patch_t, patch_h, patch_w = self.model.patch_size
        if patch_t != 1:
            raise ValueError(f"Autoregressive rollout requires temporal patch size 1, got {patch_t}.")
        return height * width // (patch_h * patch_w)

    @staticmethod
    def _sample_exit_indices(count, num_steps, device):
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                indices = torch.randint(0, num_steps, (count,), device=device)
            else:
                indices = torch.empty(count, dtype=torch.long, device=device)
            dist.broadcast(indices, src=0)
            return indices.tolist()
        return torch.randint(0, num_steps, (count,), device=device).tolist()

    @staticmethod
    def _sigma_from_timestep(timestep, dtype, scheduler):
        timesteps = scheduler.timesteps.to(
            device=timestep.device,
            dtype=torch.float32,
        )
        sigmas = scheduler.sigmas.to(device=timestep.device, dtype=dtype)
        flat = timestep.flatten().float()
        indices = torch.argmin(
            (timesteps.unsqueeze(0) - flat.unsqueeze(1)).abs(),
            dim=1,
        )
        return sigmas[indices].reshape(timestep.shape)

    @classmethod
    def _flow_to_x0(cls, sample, flow, timestep, scheduler):
        sigma = cls._sigma_from_timestep(timestep, sample.dtype, scheduler)
        sigma = sigma.reshape(
            sigma.shape[0],
            1,
            sigma.shape[1],
            *([1] * (sample.ndim - 3)),
        )
        return (sample - sigma * flow).to(dtype=sample.dtype)

    @classmethod
    def _add_noise(cls, x0, noise, timestep, scheduler):
        sigma = cls._sigma_from_timestep(timestep, x0.dtype, scheduler)
        sigma = sigma.reshape(
            sigma.shape[0],
            1,
            sigma.shape[1],
            *([1] * (x0.ndim - 3)),
        )
        return ((1.0 - sigma) * x0 + sigma * noise).to(dtype=x0.dtype)
