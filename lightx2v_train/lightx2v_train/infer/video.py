from contextlib import nullcontext
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from loguru import logger

from lightx2v_train.runtime.distributed import barrier, get_rank, get_world_size, is_distributed
from lightx2v_train.schedulers.flow_matching import CausalForcingFlowMatchScheduler
from lightx2v_train.utils.registry import INFERENCER_REGISTER

from ..model_zoo.native.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .base import BaseInferencer


def _target_hw_for_sample(sample, default_height, default_width):
    h = sample.get("target_height")
    w = sample.get("target_width")
    if h is not None and w is not None:
        return int(h), int(w)
    return default_height, default_width


@INFERENCER_REGISTER("wan_t2v_infer")
class WanT2VInferencer(BaseInferencer):
    @torch.no_grad()
    def infer(self):
        samples = self.dataloader_eval.dataset.samples
        prompts = [sample["prompt"] for sample in samples]
        rank = get_rank()
        world_size = get_world_size()

        default_height = self.infer_config.get("default_height", self.infer_config.get("height", 480))
        default_width = self.infer_config.get("default_width", self.infer_config.get("width", 832))
        num_inference_steps = self.infer_config.get("num_inference_steps", 50)
        fps = self.infer_config.get("fps", 16)
        video_quality = self.infer_config.get("video_quality", 6.0)
        macro_block_size = self.infer_config.get("macro_block_size", 16)

        logging_config = self.config.get("logging", {})
        infer_log_every_steps = max(1, int(logging_config.get("infer_log_every_steps", 10)))

        base_seed = self.infer_config.get("seed", 42)

        lora_config = self.infer_config.get("lora_config", None)
        lora_path = lora_config.get("path", None) if lora_config else None
        should_load_lora = lora_path and getattr(self.model, "_infer_lora_adapter_name", None) is None
        if should_load_lora:
            self.model.load_lora_for_infer(lora_path)

        self.enable_cfg = self.infer_config.get("enable_cfg", True)
        if self.enable_cfg:
            self.guidance_scale = self.infer_config.get("cfg_guidance_scale", 5.0)
            negative_prompt = self.infer_config.get("negative_prompt", " ")
            neg_cond = self.model.encode_condition({"prompt": negative_prompt})
        else:
            self.guidance_scale = None
            neg_cond = None

        saved_paths = []
        self.model.set_denoiser_eval()
        num_slots = (len(prompts) + world_size - 1) // world_size if is_distributed() else len(prompts)
        logger.info("[infer] start samples={} steps={} output_dir={}", len(prompts), num_inference_steps, self.output_infer_dir)
        with torch.no_grad():
            for slot in range(num_slots):
                i = slot * world_size + rank if is_distributed() else slot
                has_sample = i < len(prompts)
                prompt = prompts[i] if has_sample else " "
                sample = samples[i] if has_sample else {}

                height, width = _target_hw_for_sample(sample, default_height, default_width)
                seed = base_seed + i if has_sample else base_seed
                generator = torch.Generator(device=self.model.device).manual_seed(seed)
                pos_cond = self.model.encode_condition({"prompt": prompt})
                latent = self.model.prepare_infer_latents(height, width, generator)
                latent_hw = (latent.shape[-2], latent.shape[-1])
                self.scheduler.set_timesteps(num_inference_steps, latent_hw=latent_hw)
                total_steps = len(self.scheduler.infer_timesteps)

                if has_sample:
                    logger.info("[infer] sample={}/{} seed={} size={}x{} start", i + 1, len(prompts), seed, height, width)
                for step_idx, _ in enumerate(self.scheduler.infer_timesteps):
                    sigma = self.scheduler.infer_sigmas[step_idx].unsqueeze(0)
                    model_output = self.cfg_guided_denoise(
                        latents=latent,
                        timestep_or_sigma=sigma,
                        pos_cond=pos_cond,
                        neg_cond=neg_cond,
                    )
                    latent = self.scheduler.step(model_output, step_idx, latent)
                    step = step_idx + 1
                    if has_sample and (step == 1 or step % infer_log_every_steps == 0 or step == total_steps):
                        logger.info("[infer] sample={}/{} step={}/{}", i + 1, len(prompts), step, total_steps)

                if not has_sample:
                    continue

                videos = self.model.decode_latent(latent)

                if self.output_infer_dir is not None:
                    save_path = Path(self.output_infer_dir) / f"{i:05d}.mp4"
                    export_to_video(
                        videos[0],
                        str(save_path),
                        fps=fps,
                        quality=video_quality,
                        macro_block_size=macro_block_size,
                    )
                    logger.info("[infer] sample={}/{} saved path={}", i + 1, len(prompts), save_path)
                    saved_paths.append(str(save_path))
                logger.info("[infer] sample={}/{} done", i + 1, len(prompts))

        barrier()

        if should_load_lora:
            self.model.unload_lora_for_infer()

        saved_count = len(saved_paths)
        if is_distributed():
            saved_count_tensor = torch.tensor(saved_count, device=self.model.device, dtype=torch.int64)
            torch.distributed.all_reduce(saved_count_tensor, op=torch.distributed.ReduceOp.SUM)
            saved_count = saved_count_tensor.item()
        logger.info("[infer] finished saved={}", saved_count)
        return saved_paths


@INFERENCER_REGISTER("wan_t2v_ar_infer")
class WanT2VARInferencer(BaseInferencer):
    @torch.no_grad()
    def infer(self):
        samples = self.dataloader_eval.dataset.samples
        prompts = [sample["prompt"] for sample in samples]
        rank = get_rank()
        world_size = get_world_size()

        default_height = self.infer_config.get("default_height", self.infer_config.get("height", 480))
        default_width = self.infer_config.get("default_width", self.infer_config.get("width", 832))
        num_inference_steps = self.infer_config.get("num_inference_steps", 50)
        fps = self.infer_config.get("fps", 16)
        video_quality = self.infer_config.get("video_quality", 6.0)
        macro_block_size = self.infer_config.get("macro_block_size", 16)
        enable_cfg = self.infer_config.get("enable_cfg", True)
        guidance_scale = self.infer_config.get("cfg_guidance_scale", 3.0)
        negative_prompt = self.infer_config.get("negative_prompt", " ")
        base_seed = self.infer_config.get("seed", 42)

        lora_config = self.infer_config.get("lora_config", None)
        lora_path = lora_config.get("path", None) if lora_config else None
        should_load_lora = lora_path and getattr(self.model, "_infer_lora_adapter_name", None) is None
        if should_load_lora:
            self.model.load_lora_for_infer(lora_path)

        saved_paths = []
        self.model.set_denoiser_eval()
        num_slots = (len(prompts) + world_size - 1) // world_size if is_distributed() else len(prompts)
        logger.info(
            "[ar-infer] start samples={} steps={} chunk={} output_dir={}",
            len(prompts),
            num_inference_steps,
            self._num_frame_per_chunk(),
            self.output_infer_dir,
        )
        with torch.no_grad():
            for slot in range(num_slots):
                i = slot * world_size + rank if is_distributed() else slot
                has_sample = i < len(prompts)
                prompt = prompts[i] if has_sample else " "
                sample = samples[i] if has_sample else {}

                height, width = _target_hw_for_sample(sample, default_height, default_width)
                seed = base_seed + i if has_sample else base_seed
                generator = torch.Generator(device=self.model.device).manual_seed(seed)
                pos_cond = self.model.encode_condition({"prompt": prompt})
                neg_cond = self.model.encode_condition({"prompt": negative_prompt}) if enable_cfg else None
                latent = self.model.prepare_infer_latents(height, width, generator)

                if has_sample:
                    logger.info("[ar-infer] sample={}/{} seed={} size={}x{} start", i + 1, len(prompts), seed, height, width)
                latent = self._ar_rollout(
                    noise=latent,
                    pos_cond=pos_cond,
                    neg_cond=neg_cond,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale if enable_cfg else None,
                )

                if not has_sample:
                    continue

                videos = self.model.decode_latent(latent)
                if self.output_infer_dir is not None:
                    save_path = Path(self.output_infer_dir) / f"{i:05d}.mp4"
                    export_to_video(
                        videos[0],
                        str(save_path),
                        fps=fps,
                        quality=video_quality,
                        macro_block_size=macro_block_size,
                    )
                    logger.info("[ar-infer] sample={}/{} saved path={}", i + 1, len(prompts), save_path)
                    saved_paths.append(str(save_path))
                logger.info("[ar-infer] sample={}/{} done", i + 1, len(prompts))

        barrier()

        if should_load_lora:
            self.model.unload_lora_for_infer()

        saved_count = len(saved_paths)
        if is_distributed():
            saved_count_tensor = torch.tensor(saved_count, device=self.model.device, dtype=torch.int64)
            torch.distributed.all_reduce(saved_count_tensor, op=torch.distributed.ReduceOp.SUM)
            saved_count = saved_count_tensor.item()
        logger.info("[ar-infer] finished saved={}", saved_count)
        return saved_paths

    def _ar_rollout(self, noise, pos_cond, neg_cond, num_inference_steps, guidance_scale):
        transformer = self.model.denoiser_module()
        if not hasattr(transformer, "_forward_inference"):
            raise RuntimeError("wan_t2v_ar_infer requires the causal Wan transformer.")

        chunk_size = self._num_frame_per_chunk()
        batch_size, _, num_frames, _, _ = noise.shape
        if num_frames % chunk_size != 0:
            raise ValueError(f"AR inference latent frames={num_frames} must be divisible by chunk_size={chunk_size}.")

        output = torch.zeros_like(noise)
        frame_seq_length = self._frame_seq_length(noise)
        kv_cache_pos, crossattn_cache_pos = self._new_caches(batch_size, noise.dtype, noise.device, num_frames, frame_seq_length)
        if neg_cond is not None:
            kv_cache_neg, crossattn_cache_neg = self._new_caches(batch_size, noise.dtype, noise.device, num_frames, frame_seq_length)
        else:
            kv_cache_neg, crossattn_cache_neg = None, None
        pos_context = self.model._condition_to_context_tensor(pos_cond, batch_size=batch_size)
        neg_context = self.model._condition_to_context_tensor(neg_cond, batch_size=batch_size) if neg_cond is not None else None
        denoising_steps = self._build_ar_denoising_steps(noise.device)
        if denoising_steps is not None:
            logger.info("[ar-infer] using denoising_step_list={}", [round(float(step), 4) for step in denoising_steps.detach().cpu()])

        cache_start_frame = 0
        num_blocks = num_frames // chunk_size
        for block_idx in range(num_blocks):
            current_noise = noise[:, :, cache_start_frame : cache_start_frame + chunk_size]
            latents = current_noise
            logger.info("[ar-infer] block={}/{} frames={}..{}", block_idx + 1, num_blocks, cache_start_frame, cache_start_frame + chunk_size - 1)

            if denoising_steps is None:
                sample_scheduler = self._build_cf_unipc_scheduler(noise.device, num_inference_steps)
                for step_idx, timestep in enumerate(sample_scheduler.timesteps):
                    timestep = timestep.float().reshape(1, 1).expand(batch_size, chunk_size).to(device=noise.device)
                    flow_pred = self._predict_causal_flow(
                        latents,
                        timestep,
                        pos_context,
                        neg_context,
                        kv_cache_pos,
                        crossattn_cache_pos,
                        kv_cache_neg,
                        crossattn_cache_neg,
                        guidance_scale,
                        current_start=cache_start_frame * frame_seq_length,
                        cache_start=cache_start_frame * frame_seq_length,
                    )
                    latents = sample_scheduler.step(flow_pred, sample_scheduler.timesteps[step_idx], latents, return_dict=False)[0]
            else:
                for step_idx, current_timestep in enumerate(denoising_steps):
                    timestep = torch.full((batch_size, chunk_size), float(current_timestep), device=noise.device, dtype=torch.float32)
                    flow_pred = self._predict_causal_flow(
                        latents,
                        timestep,
                        pos_context,
                        neg_context,
                        kv_cache_pos,
                        crossattn_cache_pos,
                        kv_cache_neg,
                        crossattn_cache_neg,
                        guidance_scale,
                        current_start=cache_start_frame * frame_seq_length,
                        cache_start=cache_start_frame * frame_seq_length,
                    )
                    x0 = self._flow_to_x0(latents, flow_pred, timestep)
                    if step_idx < len(denoising_steps) - 1:
                        next_timestep = torch.full((batch_size, chunk_size), float(denoising_steps[step_idx + 1]), device=noise.device, dtype=torch.float32)
                        latents = self._add_noise_by_timestep(x0, torch.randn_like(x0), next_timestep)
                    else:
                        latents = x0

            output[:, :, cache_start_frame : cache_start_frame + chunk_size] = latents

            timestep_zero = torch.zeros((batch_size, chunk_size), device=noise.device, dtype=torch.float32)
            self._forward_causal_chunk(
                latents,
                timestep_zero,
                pos_context,
                kv_cache_pos,
                crossattn_cache_pos,
                current_start=cache_start_frame * frame_seq_length,
                cache_start=cache_start_frame * frame_seq_length,
            )
            if neg_context is not None:
                self._forward_causal_chunk(
                    latents,
                    timestep_zero,
                    neg_context,
                    kv_cache_neg,
                    crossattn_cache_neg,
                    current_start=cache_start_frame * frame_seq_length,
                    cache_start=cache_start_frame * frame_seq_length,
                )
            cache_start_frame += chunk_size

        return output

    def _predict_causal_flow(
        self,
        latents,
        timestep,
        pos_context,
        neg_context,
        kv_cache_pos,
        crossattn_cache_pos,
        kv_cache_neg,
        crossattn_cache_neg,
        guidance_scale,
        current_start,
        cache_start,
    ):
        flow_pred_cond = self._forward_causal_chunk(
            latents,
            timestep,
            pos_context,
            kv_cache_pos,
            crossattn_cache_pos,
            current_start=current_start,
            cache_start=cache_start,
        )
        if neg_context is None:
            return flow_pred_cond
        flow_pred_uncond = self._forward_causal_chunk(
            latents,
            timestep,
            neg_context,
            kv_cache_neg,
            crossattn_cache_neg,
            current_start=current_start,
            cache_start=cache_start,
        )
        return flow_pred_uncond + guidance_scale * (flow_pred_cond - flow_pred_uncond)

    def _forward_causal_chunk(self, latents, timestep, context, kv_cache, crossattn_cache, current_start, cache_start):
        transformer = self.model.denoiser_module()
        seq_len = self.model._sequence_length(latents)
        forward_context = self.model.transformer_forward_context() if hasattr(self.model, "transformer_forward_context") else nullcontext()
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
            )

    def _build_cf_unipc_scheduler(self, device, num_inference_steps):
        scheduler_config = self.config.get("scheduler", {})
        shift = self.infer_config.get("timestep_shift")
        if shift is None:
            time_shift_settings = scheduler_config.get("time_shift_settings", {})
            shift = time_shift_settings.get("time_shift_mu", 5.0)
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=scheduler_config.get("num_train_timesteps", 1000),
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(num_inference_steps, device=device, shift=float(shift))
        return scheduler

    def _configured_denoising_step_list(self):
        dmd_config = self.config.get("training", {}).get("dmd", {})
        return self.infer_config.get("denoising_step_list", dmd_config.get("denoising_step_list"))

    def _build_ar_denoising_steps(self, device):
        denoising_step_list = self._configured_denoising_step_list()
        if not denoising_step_list:
            return None
        scheduler = self._causal_forcing_scheduler()
        raw_steps = torch.tensor(denoising_step_list, dtype=torch.long, device=device)
        warp = self.infer_config.get("warp_denoising_step", self.config.get("training", {}).get("dmd", {}).get("warp_denoising_step", True))
        if not warp:
            return raw_steps.to(dtype=torch.float32)
        timesteps = torch.cat(
            [
                scheduler.timesteps.to(device=device, dtype=torch.float32),
                torch.zeros(1, device=device, dtype=torch.float32),
            ]
        )
        return timesteps[scheduler.num_train_timesteps - raw_steps]

    def _causal_forcing_scheduler(self):
        scheduler = getattr(self, "_ar_cf_scheduler", None)
        if scheduler is not None:
            return scheduler
        scheduler_config = self.config.get("scheduler", {})
        self._ar_cf_scheduler = CausalForcingFlowMatchScheduler(
            num_train_timesteps=scheduler_config.get("num_train_timesteps", 1000),
            time_shift_settings=scheduler_config.get("time_shift_settings", {}),
        )
        return self._ar_cf_scheduler

    def _sigma_from_timestep(self, timestep, dtype):
        scheduler = self._causal_forcing_scheduler()
        timesteps = scheduler.timesteps.to(device=timestep.device, dtype=torch.float32)
        sigmas = scheduler.sigmas.to(device=timestep.device, dtype=dtype)
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

    def _new_caches(self, batch_size, dtype, device, num_frames, frame_seq_length):
        transformer = self.model.denoiser_module()
        num_layers = getattr(transformer, "num_layers", None)
        if num_layers is None:
            num_layers = len(transformer.blocks)
        num_layers = int(num_layers)
        num_heads = int(transformer.num_heads)
        head_dim = int(transformer.dim // transformer.num_heads)
        local_attn_size = int(getattr(transformer, "local_attn_size", -1))
        if local_attn_size == -1:
            kv_cache_size = num_frames * frame_seq_length
        else:
            kv_cache_size = local_attn_size * frame_seq_length

        kv_cache = []
        crossattn_cache = []
        for _ in range(num_layers):
            kv_cache.append(
                {
                    "k": torch.zeros((batch_size, kv_cache_size, num_heads, head_dim), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, kv_cache_size, num_heads, head_dim), dtype=dtype, device=device),
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
            raise ValueError(f"wan_t2v_ar_infer expects temporal patch size 1, got {patch_t}.")
        return latent_height * latent_width // (patch_h * patch_w)

    def _num_frame_per_chunk(self):
        return int(getattr(self.model, "num_frame_per_chunk", self.config.get("training", {}).get("teacher_forcing", {}).get("num_frame_per_chunk", 1)))
