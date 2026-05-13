"""Pre-infer for Hunyuan3D shape generation.

Responsibilities:
  1. prepare_image  — run image_processor on raw PIL/path input
  2. encode_cond    — run conditioner → conditioning vectors (+ CFG uncond)
  3. prepare_latents — sample initial Gaussian noise scaled by scheduler sigma
"""

from __future__ import annotations

import copy

import torch
from diffusers.utils.torch_utils import randn_tensor
from loguru import logger


class Hunyuan3DPreInfer:
    def infer(
        self,
        weights,
        image,
        mask=None,
        do_classifier_free_guidance: bool = True,
        generator=None,
    ) -> dict:
        """Run image pre-processing, conditioning encoding, and latent init.

        Args:
            weights: Hunyuan3DShapeWeights instance
            image:   raw input (PIL Image, path string, or pre-processed tensor)
            mask:    optional mask tensor
            do_classifier_free_guidance: whether to concat uncond embeddings
            generator: torch.Generator for reproducible latent sampling

        Returns:
            dict with keys: cond, latents, batch_size
        """
        logger.debug("[Hunyuan3DPreInfer] prepare_image")
        cond_inputs = self._prepare_image(weights, image, mask)
        raw_image = cond_inputs.pop("image")

        logger.debug("[Hunyuan3DPreInfer] encode_cond")
        cond = self._encode_cond(weights, raw_image, cond_inputs, do_classifier_free_guidance)

        batch_size = raw_image.shape[0]
        logger.debug("[Hunyuan3DPreInfer] prepare_latents")
        latents = self._prepare_latents(weights, batch_size, generator)

        return {"cond": cond, "latents": latents, "batch_size": batch_size}

    # ------------------------------------------------------------------
    # Sub-steps
    # ------------------------------------------------------------------

    def _prepare_image(self, weights, image, mask=None) -> dict:
        """Run image_processor and return a dict with 'image' + extra cond keys."""
        import os

        if isinstance(image, torch.Tensor) and isinstance(mask, torch.Tensor):
            return {"image": image, "mask": mask}

        if isinstance(image, str) and not os.path.exists(image):
            raise FileNotFoundError(f"Image not found: {image}")

        if not isinstance(image, list):
            image = [image]

        outputs = [weights.image_processor(img) for img in image]
        cond_input: dict = {k: [] for k in outputs[0].keys()}
        for out in outputs:
            for k, v in out.items():
                cond_input[k].append(v)
        for k, v in cond_input.items():
            if isinstance(v[0], torch.Tensor):
                cond_input[k] = torch.cat(v, dim=0)
        return cond_input

    def _encode_cond(self, weights, image, additional_cond_inputs, do_classifier_free_guidance) -> dict:
        """Run conditioner and optionally prepend unconditional embeddings."""
        bsz = image.shape[0]
        cond = weights.conditioner(image=image, **additional_cond_inputs)

        if do_classifier_free_guidance:
            un_cond = weights.conditioner.unconditional_embedding(bsz, **additional_cond_inputs)

            def cat_recursive(a, b):
                if isinstance(a, torch.Tensor):
                    return torch.cat([a, b], dim=0).to(weights.dtype)
                return {k: cat_recursive(a[k], b[k]) for k in a.keys()}

            cond = cat_recursive(cond, un_cond)
        return cond

    def _prepare_latents(self, weights, batch_size: int, generator=None) -> torch.Tensor:
        """Sample initial latent noise, scaled by scheduler init_noise_sigma."""
        shape = (batch_size, *weights.vae.latent_shape)
        latents = randn_tensor(shape, generator=generator, device=weights.device, dtype=weights.dtype)
        latents = latents * getattr(weights.scheduler, "init_noise_sigma", 1.0)
        return latents
