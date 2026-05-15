from pathlib import Path

import torch
from tqdm.auto import tqdm

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

        default_height = self.infer_config.get("default_height", 1024)
        default_width = self.infer_config.get("default_width", 1024)
        num_inference_steps = self.infer_config.get("num_inference_steps", 50)

        base_seed = self.infer_config.get("seed", 42)

        lora_config = self.infer_config.get("lora_config", None)
        lora_path = lora_config.get("path", None) if lora_config else None
        if lora_path:
            self.model.load_lora_for_infer(lora_path)

        self.scheduler.set_timesteps(num_inference_steps)

        self.enable_cfg = self.infer_config.get("enable_cfg", True)
        if self.enable_cfg:
            self.guidance_scale = self.infer_config.get("cfg_guidance_scale", 4.0)
            negative_prompt = self.infer_config.get("negative_prompt", " ")
            neg_cond = self.model.encode_condition({"prompt": negative_prompt})
        else:
            self.guidance_scale = None
            neg_cond = None

        saved_paths = []
        self.model.transformer.eval()
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                height, width = _target_hw_for_sample(samples[i], default_height, default_width)
                generator = torch.Generator(device=self.model.device).manual_seed(base_seed + i)
                pos_cond = self.model.encode_condition({"prompt": prompt})
                latent = self.model.prepare_infer_latents(height, width, generator)

                for step_idx, current_timestep in enumerate(tqdm(self.scheduler.timesteps, desc=f"[{i + 1}/{len(prompts)}] Denoising")):
                    # current_timestep is in [0, 1000]
                    sigma = self.scheduler.sigmas[step_idx].unsqueeze(0)  # shape (1,) required by diffusers
                    # sigma is in [0, 1]
                    model_output = self.cfg_guided_denoise(
                        latents=latent,
                        timestep_or_sigma=sigma,
                        pos_cond=pos_cond,
                        neg_cond=neg_cond,
                    )
                    latent = self.scheduler.step(model_output, current_timestep, latent)

                images = self.model.decode_latent(latent)

                if self.output_infer_dir is not None:
                    save_path = Path(self.output_infer_dir) / f"{i:05d}.png"
                    images[0].save(save_path)
                    print(f"Saved to {save_path}")
                    saved_paths.append(str(save_path))

        if lora_path:
            self.model.unload_lora_for_infer()

        return saved_paths
