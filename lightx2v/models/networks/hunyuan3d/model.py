"""Hunyuan3D Shape Model — LightX2V-style wrapper.

Orchestrates three inference stages in the LightX2V pre/transformer/post pattern:

  pre_infer           – image processing + conditioning encoding + latent init
  transformer_infer   – flow-matching DiT denoising loop
  post_infer          – VAE decode + marching cubes → trimesh mesh

Usage in runner:
    model = Hunyuan3DShapeModel(model_path, config, device)
    meshes = model.infer(image, num_inference_steps=50, guidance_scale=5.0, generator=generator)
"""

from __future__ import annotations

import torch
from loguru import logger

from lightx2v.utils.profiler import ProfilingContext4DebugL1

from lightx2v.models.networks.hunyuan3d.weights.shape_weights import Hunyuan3DShapeWeights
from lightx2v.models.networks.hunyuan3d.infer.pre_infer import Hunyuan3DPreInfer
from lightx2v.models.networks.hunyuan3d.infer.transformer_infer import Hunyuan3DTransformerInfer
from lightx2v.models.networks.hunyuan3d.infer.post_infer import Hunyuan3DPostInfer


class Hunyuan3DShapeModel:
    """LightX2V model wrapper for Hunyuan3D shape generation."""

    def __init__(
        self,
        model_path: str,
        config: dict,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype

        logger.info(f"[Hunyuan3DShapeModel] Loading weights from: {model_path}")
        with ProfilingContext4DebugL1("Load shape weights"):
            self.weights = Hunyuan3DShapeWeights(
                model_path=model_path,
                device=device,
                dtype=dtype,
                subfolder=config.get("shape_subfolder", "hunyuan3d-dit-v2-1"),
                use_safetensors=config.get("use_safetensors", False),
                variant=config.get("variant", "fp16"),
            )

        self.pre_infer = Hunyuan3DPreInfer()
        self.transformer_infer = Hunyuan3DTransformerInfer()
        self.post_infer = Hunyuan3DPostInfer()
        logger.info("[Hunyuan3DShapeModel] Ready.")

    @torch.inference_mode()
    def infer(
        self,
        image,
        mask=None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        generator=None,
        box_v: float = 1.01,
        octree_resolution: int = 384,
        mc_level: float = 0.0,
        num_chunks: int = 8000,
        enable_pbar: bool = True,
    ) -> list:
        """Full shape generation pipeline: image → trimesh mesh list.

        Returns:
            List of trimesh.Trimesh objects (one per batch element).
        """
        do_cfg = guidance_scale >= 0 and not (
            hasattr(self.weights.model, "guidance_embed")
            and self.weights.model.guidance_embed is True
        )

        with ProfilingContext4DebugL1("Shape pre_infer"):
            pre_out = self.pre_infer.infer(
                weights=self.weights,
                image=image,
                mask=mask,
                do_classifier_free_guidance=do_cfg,
                generator=generator,
            )

        with ProfilingContext4DebugL1("Shape transformer_infer"):
            latents = self.transformer_infer.infer(
                weights=self.weights,
                cond=pre_out["cond"],
                latents=pre_out["latents"],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                enable_pbar=enable_pbar,
            )

        with ProfilingContext4DebugL1("Shape post_infer"):
            meshes = self.post_infer.infer(
                weights=self.weights,
                latents=latents,
                box_v=box_v,
                mc_level=mc_level,
                num_chunks=num_chunks,
                octree_resolution=octree_resolution,
                enable_pbar=enable_pbar,
            )

        return meshes

    def to_cpu(self) -> None:
        self.weights.to_cpu()

    def to_cuda(self) -> None:
        self.weights.to_cuda()
