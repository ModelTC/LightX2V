"""Transformer-infer for Hunyuan3D shape generation.

Runs the flow-matching denoising loop:
  for each timestep t:
    1. (optional CFG) duplicate latents
    2. scale latent by scheduler
    3. DiT forward → noise_pred
    4. (optional CFG) merge cond / uncond predictions
    5. scheduler.step → updated latents
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm


def _retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=None):
    sigmas = np.linspace(0, 1, num_inference_steps) if sigmas is None else sigmas
    scheduler.set_timesteps(sigmas=sigmas, device=device)
    return scheduler.timesteps, len(scheduler.timesteps)


class Hunyuan3DTransformerInfer:
    def infer(
        self,
        weights,
        cond: dict,
        latents: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        sigmas=None,
        enable_pbar: bool = True,
    ) -> torch.Tensor:
        """Run flow-matching denoising and return denoised latents.

        Args:
            weights:             Hunyuan3DShapeWeights
            cond:                output of pre_infer (conditioned+uncond embeddings)
            latents:             initial noise latents from pre_infer
            num_inference_steps: number of denoising steps
            guidance_scale:      CFG scale
            sigmas:              custom sigma schedule (None → linspace 0→1)
            enable_pbar:         show tqdm progress bar

        Returns:
            Denoised latents tensor
        """
        device = weights.device
        dtype = weights.dtype

        do_cfg = guidance_scale >= 0 and not (
            hasattr(weights.model, "guidance_embed") and weights.model.guidance_embed is True
        )

        guidance = None
        if hasattr(weights.model, "guidance_embed") and weights.model.guidance_embed is True:
            batch_size = latents.shape[0]
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)

        timesteps, num_inference_steps = _retrieve_timesteps(
            weights.scheduler, num_inference_steps, device, sigmas
        )
        logger.debug(f"[Hunyuan3DTransformerInfer] steps={num_inference_steps}  CFG={do_cfg}  gs={guidance_scale}")

        for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Shape DiT sampling:")):
            latent_input = torch.cat([latents] * 2) if do_cfg else latents
            timestep = t.expand(latent_input.shape[0]).to(latents.dtype)
            timestep = timestep / weights.scheduler.config.num_train_timesteps

            noise_pred = weights.model(latent_input, timestep, cond, guidance=guidance)

            if do_cfg:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents = weights.scheduler.step(noise_pred, t, latents).prev_sample

        logger.debug("[Hunyuan3DTransformerInfer] Denoising done.")
        return latents
