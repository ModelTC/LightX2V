"""Checkpoint loading helpers for Hunyuan3D-2.1 shape models."""

from __future__ import annotations

import os
from typing import Any

import torch
import yaml
from loguru import logger


def resolve_model_dir(model_path: str, subfolder: str | None = None) -> str:
    subfolder = subfolder or "hunyuan3d-dit-v2-1"
    candidate = os.path.join(model_path, subfolder)
    if os.path.isdir(candidate):
        return candidate
    if os.path.isdir(model_path):
        return model_path
    raise FileNotFoundError(f"Could not locate Hunyuan3D weights under '{candidate}' or '{model_path}'")


def resolve_ckpt_paths(model_dir: str, use_safetensors: bool = False, variant: str = "fp16") -> tuple[str, str]:
    extension = "safetensors" if use_safetensors else "ckpt"
    variant_suffix = "" if variant is None else f".{variant}"
    ckpt_name = f"model{variant_suffix}.{extension}"
    config_path = os.path.join(model_dir, "config.yaml")
    ckpt_path = os.path.join(model_dir, ckpt_name)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing config.yaml: {config_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    return config_path, ckpt_path


def load_pipeline_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint_dict(ckpt_path: str, use_safetensors: bool = False) -> dict[str, dict[str, torch.Tensor]]:
    logger.info(f"Loading Hunyuan3D checkpoint from {ckpt_path}")
    if use_safetensors:
        import safetensors.torch

        safetensors_ckpt = safetensors.torch.load_file(ckpt_path, device="cpu")
        ckpt: dict[str, dict[str, torch.Tensor]] = {}
        for key, value in safetensors_ckpt.items():
            model_name, new_key = key.split(".", 1)
            ckpt.setdefault(model_name, {})[new_key] = value
        return ckpt

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return raw
