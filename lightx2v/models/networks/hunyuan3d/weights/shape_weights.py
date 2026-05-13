"""Weights holder for Hunyuan3D shape generation (conditioner + DiT + ShapeVAE).

All three sub-models are loaded from a single checkpoint bundle via
Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(), so there is no
separate pre/transformer/post weight file — they are unified here.

Attributes after load:
    conditioner       : image conditioner (CLIP / DINOv2)
    image_processor   : image pre-processor
    model             : DiT transformer
    scheduler         : flow-matching scheduler
    vae               : ShapeVAE (decodes latents → mesh)
    device            : torch.device
    dtype             : torch.dtype
"""

from __future__ import annotations

import torch
from loguru import logger

from lightx2v.models.networks.hunyuan3d.hy3dshape.models.autoencoders import (
    ShapeVAE,
    SurfaceExtractors,
)
from lightx2v.models.networks.hunyuan3d.hy3dshape.utils import smart_load_model
from lightx2v.models.networks.hunyuan3d.hy3dshape.utils.misc import instantiate_from_config


class Hunyuan3DShapeWeights:
    """Holds all sub-model instances for Hunyuan3D shape generation."""

    def __init__(
        self,
        model_path: str,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        subfolder: str = "hunyuan3d-dit-v2-1",
        use_safetensors: bool = False,
        variant: str = "fp16",
    ) -> None:
        import yaml

        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = dtype

        logger.info(f"[Hunyuan3DShapeWeights] Loading from: {model_path}  subfolder={subfolder}")
        config_path, ckpt_path = smart_load_model(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant,
        )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if use_safetensors:
            import safetensors.torch
            raw = safetensors.torch.load_file(ckpt_path, device="cpu")
            ckpt: dict = {}
            for key, value in raw.items():
                model_name, new_key = key.split(".", 1)
                ckpt.setdefault(model_name, {})[new_key] = value
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        self.model = instantiate_from_config(config["model"])
        self.model.load_state_dict(ckpt["model"])

        self.vae = instantiate_from_config(config["vae"])
        self.vae.load_state_dict(ckpt["vae"], strict=False)

        self.conditioner = instantiate_from_config(config["conditioner"])
        if "conditioner" in ckpt:
            self.conditioner.load_state_dict(ckpt["conditioner"])

        self.image_processor = instantiate_from_config(config["image_processor"])
        self.scheduler = instantiate_from_config(config["scheduler"])

        self._move_to_device()
        logger.info("[Hunyuan3DShapeWeights] All sub-models loaded.")

    def _move_to_device(self) -> None:
        for attr in ("model", "vae", "conditioner"):
            getattr(self, attr).to(device=self.device, dtype=self.dtype)

    def to(self, device=None, dtype=None) -> "Hunyuan3DShapeWeights":
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = torch.device(device)
        self._move_to_device()
        return self

    def to_cpu(self) -> None:
        self.to(device="cpu")

    def to_cuda(self) -> None:
        self.to(device=self.device)
