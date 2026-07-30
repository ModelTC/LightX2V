import torch
import torch.distributed as dist

from lightx2v_train.runtime.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    is_sequence_parallel_enabled,
)
from lightx2v_train.runtime.sequence_parallel import all_gather_sequence
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .video_trainer import VideoDmdTrainer


@TRAINER_REGISTER("video_ar_dmd")
class VideoArDmdTrainer(VideoDmdTrainer):
    trainer_name = "video_ar_dmd"
    allowed_model_names = {"wan_t2v_ar", "wan_t2v_14b_ar", "wan_ti2v_5b", "wan_ti2v_5b_ar"}
    supports_diversity_loss = False
    supports_real_data_fake = False

    def __init__(self, config):
        super().__init__(config)
        self.num_frame_per_chunk = int(self.dmd_config["num_frame_per_chunk"] if "num_frame_per_chunk" in self.dmd_config else self.model_config.get("num_frame_per_chunk", 3))
        self.same_step_across_blocks = bool(self.dmd_config.get("same_step_across_blocks", True))
        self.context_noise = float(self.dmd_config.get("context_noise", 0.0))
        self.sequence_parallel_cache = bool(self.dmd_config["sp_cache"] if "sp_cache" in self.dmd_config else False)

    def run_back_simulation(self, condition, latent_shape, grad_enabled, xt=None):
        transformer = self.model.denoiser_module()
        if not hasattr(transformer, "_forward_inference"):
            raise RuntimeError("ar_dmd requires the causal Wan transformer.")

        if xt is None:
            xt = self.sample_initial_latents(latent_shape)
        batch_size, _, num_frames, _, _ = xt.shape
        if num_frames % self.num_frame_per_chunk != 0:
            raise ValueError(f"ar_dmd latent frames={num_frames} must be divisible by num_frame_per_chunk={self.num_frame_per_chunk}.")

        use_sp_cache = self._use_sequence_parallel_cache()
        sp_size = get_sequence_parallel_world_size() if use_sp_cache else 1
        sp_rank = get_sequence_parallel_rank() if use_sp_cache else 0
        if use_sp_cache and self.num_frame_per_chunk % sp_size != 0:
            raise ValueError(f"training.dmd.num_frame_per_chunk={self.num_frame_per_chunk} must be divisible by sequence_parallel.size={sp_size} when training.dmd.sp_cache=true.")

        self.model.transformer.train()
        output_chunks = []
        context = self.model._condition_to_context_tensor(condition, batch_size=batch_size)
        frame_seq_length = self._frame_seq_length(xt)
        kv_cache, crossattn_cache = self._new_caches(batch_size, xt.dtype, xt.device, num_frames, frame_seq_length, sequence_parallel_cache=use_sp_cache)
        num_blocks = num_frames // self.num_frame_per_chunk
        exit_indices = self._sample_exit_indices(num_blocks, len(self.denoising_steps), xt.device)

        current_start_frame = 0
        for block_idx in range(num_blocks):
            current_num_frames = self.num_frame_per_chunk
            if use_sp_cache:
                local_num_frames = current_num_frames // sp_size
                local_start_frame = current_start_frame + sp_rank * local_num_frames
            else:
                local_num_frames = current_num_frames
                local_start_frame = current_start_frame
            latents = xt[:, :, local_start_frame : local_start_frame + local_num_frames]
            exit_idx = int(exit_indices[0] if self.same_step_across_blocks else exit_indices[block_idx])

            x0 = None
            for step_idx, current_timestep in enumerate(self.denoising_steps):
                timestep = torch.full(
                    (batch_size, local_num_frames),
                    float(current_timestep),
                    device=xt.device,
                    dtype=torch.float32,
                )
                enable_step_grad = grad_enabled and step_idx == exit_idx
                context_mgr = torch.enable_grad if enable_step_grad else torch.no_grad
                with context_mgr():
                    flow_pred = self._forward_causal_chunk(
                        self.model,
                        latents,
                        timestep,
                        context,
                        kv_cache,
                        crossattn_cache,
                        current_start=current_start_frame * frame_seq_length,
                        cache_start=current_start_frame * frame_seq_length,
                        local_frame_offset=local_start_frame,
                        balanced_sequence_parallel=use_sp_cache,
                    )
                    x0 = self._flow_to_x0(latents, flow_pred, timestep)

                if step_idx == exit_idx:
                    break

                next_timestep = torch.full(
                    (batch_size, local_num_frames),
                    float(self.denoising_steps[step_idx + 1]),
                    device=xt.device,
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    latents = self._add_noise_by_timestep(x0, torch.randn_like(x0), next_timestep)

            output_chunks.append(all_gather_sequence(x0, dim=2) if use_sp_cache else x0)

            cache_latents = x0.detach()
            cache_timestep = torch.full(
                (batch_size, local_num_frames),
                self.context_noise,
                device=xt.device,
                dtype=torch.float32,
            )
            if self.context_noise > 0:
                cache_latents = self._add_noise_by_timestep(cache_latents, torch.randn_like(cache_latents), cache_timestep)
            with torch.no_grad():
                self._forward_causal_chunk(
                    self.model,
                    cache_latents,
                    cache_timestep,
                    context,
                    kv_cache,
                    crossattn_cache,
                    current_start=current_start_frame * frame_seq_length,
                    cache_start=current_start_frame * frame_seq_length,
                    local_frame_offset=local_start_frame,
                    balanced_sequence_parallel=use_sp_cache,
                )

            current_start_frame += current_num_frames

        return (
            torch.cat(output_chunks, dim=2).to(dtype=self.running_dtype),
            *self._denoised_timestep_window(exit_indices),
        )

    def _use_sequence_parallel_cache(self):
        enabled = bool(self.sequence_parallel_cache and is_sequence_parallel_enabled())
        if enabled and bool(getattr(self.model.denoiser_module(), "defer_kv_cache_updates", False)):
            raise ValueError("training.dmd.sp_cache=true does not support model.defer_kv_cache_updates=true. Set model.defer_kv_cache_updates=false or disable training.dmd.sp_cache.")
        return enabled

    def _denoised_timestep_window(self, exit_indices):
        if not self.same_step_across_blocks:
            return None, None

        exit_idx = int(exit_indices[0])
        denoised_timestep_from = self._raw_timestep_from_warped_step(self.denoising_steps[exit_idx])
        if exit_idx == len(self.denoising_steps) - 1:
            denoised_timestep_to = 0
        else:
            denoised_timestep_to = self._raw_timestep_from_warped_step(self.denoising_steps[exit_idx + 1])
        return denoised_timestep_from, denoised_timestep_to

    def _sample_exit_indices(self, num_blocks, num_steps, device):
        count = 1 if self.same_step_across_blocks else num_blocks
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                indices = torch.randint(0, num_steps, (count,), device=device)
            else:
                indices = torch.empty(count, dtype=torch.long, device=device)
            dist.broadcast(indices, src=0)
            return indices.tolist()
        return torch.randint(0, num_steps, (count,), device=device).tolist()

    def _forward_causal_chunk(
        self,
        model,
        latents,
        timestep,
        context,
        kv_cache,
        crossattn_cache,
        current_start,
        cache_start,
        local_frame_offset=0,
        balanced_sequence_parallel=False,
    ):
        transformer = model.denoiser_module()
        seq_len = model._sequence_length(latents)
        forward_context = model.transformer_forward_context() if hasattr(model, "transformer_forward_context") else torch.no_grad()
        with forward_context:
            return transformer(
                latents,
                t=timestep,
                context=context,
                seq_len=seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                local_frame_offset=local_frame_offset,
                balanced_sequence_parallel=balanced_sequence_parallel,
            )

    def _new_caches(self, batch_size, dtype, device, num_frames, frame_seq_length, sequence_parallel_cache=False):
        transformer = self.model.denoiser_module()
        num_layers = int(getattr(transformer, "num_layers", len(transformer.blocks)))
        num_heads = int(transformer.num_heads)
        head_dim = int(transformer.dim // transformer.num_heads)
        kv_num_heads = num_heads
        if sequence_parallel_cache:
            sp_size = get_sequence_parallel_world_size()
            if num_heads % sp_size != 0:
                raise ValueError(f"transformer.num_heads={num_heads} must be divisible by sequence_parallel.size={sp_size}.")
            kv_num_heads = num_heads // sp_size
        local_attn_size = int(getattr(transformer, "local_attn_size", -1))
        kv_cache_size = num_frames * frame_seq_length if local_attn_size == -1 else local_attn_size * frame_seq_length

        kv_cache = []
        crossattn_cache = []
        for _ in range(num_layers):
            kv_cache.append(
                {
                    "k": torch.zeros((batch_size, kv_cache_size, kv_num_heads, head_dim), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, kv_cache_size, kv_num_heads, head_dim), dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )
            crossattn_cache.append(
                {
                    "k": torch.zeros((batch_size, self.model.max_sequence_length, num_heads, head_dim), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, self.model.max_sequence_length, num_heads, head_dim), dtype=dtype, device=device),
                    "is_init": False,
                }
            )
        return kv_cache, crossattn_cache

    def _frame_seq_length(self, latent):
        _, _, _, latent_height, latent_width = latent.shape
        patch_t, patch_h, patch_w = self.model.patch_size
        if patch_t != 1:
            raise ValueError(f"ar_dmd expects temporal patch size 1, got {patch_t}.")
        return latent_height * latent_width // (patch_h * patch_w)

    def _sigma_from_timestep(self, timestep, dtype):
        timesteps = self.denoising_scheduler.timesteps.to(device=timestep.device, dtype=torch.float32)
        sigmas = self.denoising_scheduler.sigmas.to(device=timestep.device, dtype=dtype)
        flat_timestep = timestep.flatten().float()
        index = torch.argmin((timesteps.unsqueeze(0) - flat_timestep.unsqueeze(1)).abs(), dim=1)
        return sigmas[index].reshape(timestep.shape)

    def _expand_frame_sigma(self, sigma, ndim):
        return sigma.reshape(sigma.shape[0], 1, sigma.shape[1], *([1] * (ndim - 3)))

    def _flow_to_x0(self, xt, flow_pred, timestep):
        sigma = self._sigma_from_timestep(timestep, xt.dtype)
        sigma = self._expand_frame_sigma(sigma, xt.ndim)
        return (xt - sigma * flow_pred).to(dtype=xt.dtype)

    def _add_noise_by_timestep(self, x0, noise, timestep):
        sigma = self._sigma_from_timestep(timestep, x0.dtype)
        sigma = self._expand_frame_sigma(sigma, x0.ndim)
        return ((1.0 - sigma) * x0 + sigma * noise).to(dtype=x0.dtype)
