import json
import os

import torch.distributed as dist
from loguru import logger
from torch.distributed.tensor.device_mesh import init_device_mesh


def get_default_config():
    default_config = {
        "do_mm_calib": False,
        "cpu_offload": False,
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
        "parallel": False,
        "seq_parallel": False,
        "cfg_parallel": False,
        "enable_cfg": False,
        "use_image_encoder": True,
        "lat_h": None,
        "lat_w": None,
        "tgt_h": None,
        "tgt_w": None,
        "target_shape": None,
        "return_video": False,
        "audio_num": None,
        "person_num": None,
    }
    return default_config


def set_config(args):
    config = get_default_config()
    config.update({k: v for k, v in vars(args).items()})

    with open(config["config_json"], "r") as f:
        config_json = json.load(f)
    config.update(config_json)

    if os.path.exists(os.path.join(config["model_path"], "config.json")):
        with open(os.path.join(config["model_path"], "config.json"), "r") as f:
            model_config = json.load(f)
        config.update(model_config)
    elif os.path.exists(os.path.join(config["model_path"], "low_noise_model", "config.json")):  # 需要一个更优雅的update方法
        with open(os.path.join(config["model_path"], "low_noise_model", "config.json"), "r") as f:
            model_config = json.load(f)
        config.update(model_config)
    elif os.path.exists(os.path.join(config["model_path"], "distill_models", "low_noise_model", "config.json")):  # 需要一个更优雅的update方法
        with open(os.path.join(config["model_path"], "distill_models", "low_noise_model", "config.json"), "r") as f:
            model_config = json.load(f)
        config.update(model_config)
    elif os.path.exists(os.path.join(config["model_path"], "original", "config.json")):
        with open(os.path.join(config["model_path"], "original", "config.json"), "r") as f:
            model_config = json.load(f)
        config.update(model_config)
    # load quantized config
    if config.get("dit_quantized_ckpt", None) is not None:
        config_path = os.path.join(config["dit_quantized_ckpt"], "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                model_config = json.load(f)
            config.update(model_config)

    if config["task"] == "i2v":
        if config["target_video_length"] % config["vae_stride"][0] != 1:
            logger.warning(f"`num_frames - 1` has to be divisible by {config['vae_stride'][0]}. Rounding to the nearest number.")
            config["target_video_length"] = config["target_video_length"] // config["vae_stride"][0] * config["vae_stride"][0] + 1

    if config["audio_path"]:
        if os.path.isdir(config["audio_path"]):
            logger.info(f"audio_path is a directory, loading config.json from {config['audio_path']}")
            audio_config_path = os.path.join(config["audio_path"], "config.json")
            assert os.path.exists(audio_config_path), "config.json not found in audio_path"
            with open(audio_config_path, "r") as f:
                audio_config = json.load(f)
            for talk_object in audio_config["talk_objects"]:
                talk_object["audio"] = os.path.join(config["audio_path"], talk_object["audio"])
                talk_object["mask"] = os.path.join(config["audio_path"], talk_object["mask"])
            config.update(audio_config)
        else:
            logger.info(f"audio_path is a file: {config['audio_path']}")

    assert not (config["save_video_path"] and config["return_video"]), "save_video_path and return_video cannot be set at the same time"

    return config


def set_parallel_config(config):
    if config["parallel"]:
        cfg_p_size = config["parallel"].get("cfg_p_size", 1)
        seq_p_size = config["parallel"].get("seq_p_size", 1)
        assert cfg_p_size * seq_p_size == dist.get_world_size(), f"cfg_p_size * seq_p_size must be equal to world_size"
        config["device_mesh"] = init_device_mesh("cuda", (cfg_p_size, seq_p_size), mesh_dim_names=("cfg_p", "seq_p"))

        if config["parallel"] and config["parallel"].get("seq_p_size", False) and config["parallel"]["seq_p_size"] > 1:
            config["seq_parallel"] = True

        if config.get("enable_cfg", False) and config["parallel"] and config["parallel"].get("cfg_p_size", False) and config["parallel"]["cfg_p_size"] > 1:
            config["cfg_parallel"] = True


def print_config(config):
    config_to_print = config.copy()
    config_to_print.pop("device_mesh", None)
    if config["parallel"]:
        if dist.get_rank() == 0:
            logger.info(f"config:\n{json.dumps(config_to_print, ensure_ascii=False, indent=4)}")
    else:
        logger.info(f"config:\n{json.dumps(config_to_print, ensure_ascii=False, indent=4)}")
