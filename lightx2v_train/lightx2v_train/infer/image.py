from pathlib import Path

import torch
from loguru import logger

from lightx2v_train.runtime.distributed import barrier, get_rank, get_world_size, is_distributed
from lightx2v_train.utils.registry import INFERENCER_REGISTER

from .base import BaseInferencer


def _target_hw_for_sample(sample, default_height, default_width):
    h = sample.get("target_height")
    w = sample.get("target_width")
    if h is not None and w is not None:
        return int(h), int(w)
    return default_height, default_width


@INFERENCER_REGISTER("image_infer")
class ImageInferencer(BaseInferencer):
    @torch.no_grad()
    def infer(self):
        samples = self.dataloader_eval.dataset.samples
        prompts = [sample["prompt"] for sample in samples]
        rank = get_rank()
        world_size = get_world_size()

        default_height = self.infer_config.get("default_height", 1024)
        default_width = self.infer_config.get("default_width", 1024)
        num_inference_steps = self.infer_config.get("num_inference_steps", 50)
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
            self.guidance_scale = self.infer_config.get("cfg_guidance_scale", 4.0)
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
                latent_hw = (latent.shape[3], latent.shape[4])
                self.scheduler.set_timesteps(num_inference_steps, latent_hw=latent_hw)
                total_steps = len(self.scheduler.infer_timesteps)

                if has_sample:
                    logger.info("[infer] sample={}/{} seed={} size={}x{} start", i + 1, len(prompts), seed, height, width)
                for step_idx, _ in enumerate(self.scheduler.infer_timesteps):
                    # scheduler timesteps are in [0, 1000]
                    sigma = self.scheduler.infer_sigmas[step_idx].unsqueeze(0)  # shape (1,) required by diffusers
                    # sigma is in [0, 1]
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

                images = self.model.decode_latent(latent)

                if self.output_infer_dir is not None:
                    save_path = Path(self.output_infer_dir) / f"{i:05d}.png"
                    images[0].save(save_path)
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
