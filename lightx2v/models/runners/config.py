"""Checkpoint metadata readers shared by model configuration loaders."""

import json
import os

from loguru import logger

from lightx2v.utils.request_defaults import get_input_value, set_input_value
from lightx2v.utils.set_config import merge_config

CHECKPOINT_SUBFOLDERS = ("", "low_noise_model", "distill_models/low_noise_model", "original", "transformer")


def read_config(path):
    with open(path, "r") as f:
        return json.load(f)


def load_vae_config(deployment_config):
    path = os.path.join(deployment_config["model_path"], "vae", "config.json")
    if not os.path.exists(path):
        return {}
    vae_config = read_config(path)
    if "temperal_downsample" in vae_config:
        return {"vae_scale_factor": 2 ** len(vae_config["temperal_downsample"])}
    if "block_out_channels" in vae_config:
        return {"vae_scale_factor": 2 ** (len(vae_config["block_out_channels"]) - 1)}
    return {}


def read_checkpoint_config(model_path, subfolders=CHECKPOINT_SUBFOLDERS):
    for subfolder in subfolders:
        path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(path):
            return read_config(path)
    return {}


def load_quantized_config(deployment_config):
    model_path = deployment_config.get("dit_quantized_ckpt")
    return {} if model_path is None else read_checkpoint_config(model_path, ("",))


def load_checkpoint_config(deployment_config):
    return read_checkpoint_config(deployment_config["model_path"]) | load_quantized_config(deployment_config) | load_vae_config(deployment_config)


def normalize_num_frames(config):
    if config.get("task") not in ("i2v", "t2av", "i2av", "i2va", "s2v", "rs2v", "ltx2_s2v", "v2av"):
        return
    num_frames = get_input_value(config, "num_frames", None)
    if num_frames is None or "vae_stride" not in config:
        return
    temporal_stride = int(config["vae_stride"][0])
    if (num_frames - 1) % temporal_stride != 0:
        aligned = num_frames // temporal_stride * temporal_stride + 1
        set_input_value(config, "num_frames", aligned)
        logger.warning(f"`num_frames - 1` must be divisible by {temporal_stride}; using {aligned} instead of {num_frames}.")


def load_config(deployment_config):
    config = merge_config(load_checkpoint_config(deployment_config), deployment_config)
    normalize_num_frames(config)
    return config


def load_vae_only_config(deployment_config):
    return merge_config(load_vae_config(deployment_config), deployment_config)
