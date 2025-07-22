import json
import os
from easydict import EasyDict
from loguru import logger


def get_default_config():
    default_config = {
        "do_mm_calib": False,
        "cpu_offload": False,
        "parallel_attn_type": None,  # [None, "ulysses", "ring"]
        "parallel_vae": False,
        "max_area": False,
        "vae_stride": (4, 8, 8),
        "patch_size": (1, 2, 2),
        "feature_caching": "NoCaching",  # ["NoCaching", "TaylorSeer", "Tea"]
        "teacache_thresh": 0.26,
        "use_ret_steps": False,
        "use_bfloat16": True,
        "lora_configs": None,  # List of dicts with 'path' and 'strength' keys
        "mm_config": {},
        "use_prompt_enhancer": False,
    }
    return default_config


def set_config(args):
    config = get_default_config()
    config.update({k: v for k, v in vars(args).items()})
    config = EasyDict(config)

    with open(config.config_json, "r") as f:
        config_json = json.load(f)
    config.update(config_json)

    if os.path.exists(os.path.join(config.model_path, "config.json")):
        with open(os.path.join(config.model_path, "config.json"), "r") as f:
            model_config = json.load(f)
        config.update(model_config)

    if config.get("dit_quantized_ckpt", None) is not None:
        config_path = os.path.join(config.dit_quantized_ckpt, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                model_config = json.load(f)
            config.update(model_config)

    if config.task == "i2v":
        if config.target_video_length % config.vae_stride[0] != 1:
            logger.warning(f"`num_frames - 1` has to be divisible by {config.vae_stride[0]}. Rounding to the nearest number.")
            config.target_video_length = config.target_video_length // config.vae_stride[0] * config.vae_stride[0] + 1

    return config
