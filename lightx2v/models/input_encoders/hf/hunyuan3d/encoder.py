"""Image preprocessing and condition encoding for Hunyuan3D shape generation."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from lightx2v.models.input_encoders.hf.hunyuan3d.conditioner import SingleImageEncoder
from lightx2v.models.input_encoders.hf.hunyuan3d.image_processor import ImageProcessorV2
from lightx2v.models.input_encoders.hf.hunyuan3d.rembg import BackgroundRemover
from lightx2v.models.networks.hunyuan3d.utils.checkpoint import (
    load_checkpoint_dict,
    load_pipeline_config,
    resolve_ckpt_paths,
    resolve_model_dir,
)
from lightx2v_platform.base.global_var import AI_DEVICE


class Hunyuan3DConditionEncoder(nn.Module):
    """Wraps image processor + DINO conditioner for Hunyuan3D shape inference."""

    def __init__(self, image_processor, conditioner, device=AI_DEVICE, dtype=torch.float16):
        super().__init__()
        self.image_processor = image_processor
        self.conditioner = conditioner
        self.device = torch.device(device)
        self.dtype = dtype
        self.conditioner.to(device=device, dtype=dtype)
        self.eval()

    @classmethod
    def from_pretrained(cls, config, ckpt: dict[str, dict[str, torch.Tensor]] | None = None):
        model_path = config["model_path"]
        subfolder = config.get("subfolder", "hunyuan3d-dit-v2-1")
        use_safetensors = bool(config.get("use_safetensors", False))
        variant = config.get("variant", "fp16")
        dtype_name = str(config.get("dtype", "fp16")).lower()
        dtype = torch.float16 if dtype_name in {"fp16", "float16"} else torch.bfloat16
        device = config.get("device", AI_DEVICE)

        model_dir = resolve_model_dir(model_path, subfolder)
        config_path, ckpt_path = resolve_ckpt_paths(model_dir, use_safetensors=use_safetensors, variant=variant)
        pipeline_cfg = load_pipeline_config(config_path)
        if ckpt is None:
            ckpt = load_checkpoint_dict(ckpt_path, use_safetensors=use_safetensors)

        image_processor = ImageProcessorV2(**pipeline_cfg["image_processor"]["params"])
        conditioner = SingleImageEncoder(**pipeline_cfg["conditioner"]["params"])
        if "conditioner" in ckpt:
            conditioner.load_state_dict(ckpt["conditioner"])

        conditioner.eval()
        for param in conditioner.parameters():
            param.requires_grad = False

        encoder = cls(image_processor, conditioner, device=device, dtype=dtype)
        return encoder

    def prepare_image(self, image: Image.Image | list[Image.Image], mask=None) -> dict[str, torch.Tensor]:
        if not isinstance(image, list):
            image = [image]

        outputs = [self.image_processor(img) for img in image]
        cond_input = {k: [] for k in outputs[0].keys()}
        for output in outputs:
            for key, value in output.items():
                cond_input[key].append(value)
        for key, value in cond_input.items():
            if isinstance(value[0], torch.Tensor):
                cond_input[key] = torch.cat(value, dim=0)
        return cond_input

    def encode_cond(
        self,
        image_tensor: torch.Tensor,
        additional_cond_inputs: dict[str, Any],
        do_classifier_free_guidance: bool,
        dual_guidance: bool = False,
    ):
        bsz = image_tensor.shape[0]
        cond = self.conditioner(image=image_tensor, **additional_cond_inputs)

        if not do_classifier_free_guidance:
            return {"cond": cond}

        un_cond = self.conditioner.unconditional_embedding(bsz, **additional_cond_inputs)
        if dual_guidance:
            un_cond_drop_main = copy.deepcopy(un_cond)
            un_cond_drop_main["additional"] = cond["additional"]
            return {
                "cond": cond,
                "uncond_drop_main": un_cond_drop_main,
                "uncond": un_cond,
            }
        return {"cond": cond, "uncond": un_cond}

    @staticmethod
    def _cat_recursive(a, b, c=None, dtype=None):
        if c is None:
            if isinstance(a, torch.Tensor):
                out = torch.cat([a, b], dim=0)
                return out.to(dtype=dtype) if dtype is not None else out
            return {k: Hunyuan3DConditionEncoder._cat_recursive(a[k], b[k], dtype=dtype) for k in a.keys()}

        if isinstance(a, torch.Tensor):
            out = torch.cat([a, b, c], dim=0)
            return out.to(dtype=dtype) if dtype is not None else out
        return {k: Hunyuan3DConditionEncoder._cat_recursive(a[k], b[k], c[k], dtype=dtype) for k in a.keys()}


class Hunyuan3DImagePreprocessor:
    """Optional rembg + RGBA conversion before condition encoding."""

    def __init__(self, enable_rembg: bool = True):
        self.enable_rembg = enable_rembg
        self.rembg = BackgroundRemover() if enable_rembg else None

    def __call__(self, image_path: str) -> Image.Image:
        image = Image.open(image_path)
        if image.mode == "RGB" and self.enable_rembg:
            if self.rembg is None:
                self.rembg = BackgroundRemover()
            return self.rembg(image.convert("RGB"))
        return image.convert("RGBA")
