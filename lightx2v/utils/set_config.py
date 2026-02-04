import json
import os

import torch
import torch.distributed as dist
from loguru import logger
from torch.distributed.tensor.device_mesh import init_device_mesh

from lightx2v.utils.input_info import ALL_INPUT_INFO_KEYS
from lightx2v.utils.lockable_dict import LockableDict
from lightx2v.utils.utils import is_main_process
from lightx2v_platform.base.global_var import AI_DEVICE


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
        "use_prompt_enhancer": False,
        "parallel": False,
        "seq_parallel": False,
        "cfg_parallel": False,
        "enable_cfg": False,
        "use_image_encoder": True,
    }
    default_config = LockableDict(default_config)
    return default_config


def set_config(args):
    config = get_default_config()
    config.update({k: v for k, v in vars(args).items() if k not in ALL_INPUT_INFO_KEYS})

    if config.get("config_json", None) is not None:
        logger.info(f"Loading some config from {config['config_json']}")
        with open(config["config_json"], "r") as f:
            config_json = json.load(f)
        config.update(config_json)

    if config["model_cls"] in ["hunyuan_video_1.5", "hunyuan_video_1.5_distill"]:  # Special config for hunyuan video 1.5 model folder structure
        config["transformer_model_path"] = os.path.join(config["model_path"], "transformer", config["transformer_model_name"])  # transformer_model_name: [480p_t2v, 480p_i2v, 720p_t2v, 720p_i2v]
        if os.path.exists(os.path.join(config["transformer_model_path"], "config.json")):
            with open(os.path.join(config["transformer_model_path"], "config.json"), "r") as f:
                model_config = json.load(f)
            config.update(model_config)
    elif config["model_cls"] in ["worldplay_distill", "worldplay_ar", "worldplay_bi"]:  # Special config for WorldPlay models
        config["transformer_model_path"] = os.path.join(config["model_path"], "transformer", config["transformer_model_name"])
        if os.path.exists(os.path.join(config["transformer_model_path"], "config.json")):
            with open(os.path.join(config["transformer_model_path"], "config.json"), "r") as f:
                model_config = json.load(f)
            config.update(model_config)
    elif config["model_cls"] == "longcat_image":  # Special config for longcat_image: load both root and transformer config
        if os.path.exists(os.path.join(config["model_path"], "config.json")):
            with open(os.path.join(config["model_path"], "config.json"), "r") as f:
                model_config = json.load(f)
            config.update(model_config)
        if os.path.exists(os.path.join(config["model_path"], "transformer", "config.json")):
            with open(os.path.join(config["model_path"], "transformer", "config.json"), "r") as f:
                model_config = json.load(f)
            config.update(model_config)
    else:
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
        elif os.path.exists(os.path.join(config["model_path"], "transformer", "config.json")):
            with open(os.path.join(config["model_path"], "transformer", "config.json"), "r") as f:
                model_config = json.load(f)

            if config["model_cls"] == "z_image":
                # https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/transformer/config.json
                z_image_patch_size = model_config.pop("all_patch_size", [2])
                z_image_f_patch_size = model_config.pop("all_f_patch_size", [1])
                if not (len(z_image_patch_size) == 1 and len(z_image_f_patch_size) == 1):
                    raise ValueError(
                        f"Expected 'all_patch_size' and 'all_f_patch_size' in z_image config to be lists of length 1, "
                        f"but got lengths {len(z_image_patch_size)} and {len(z_image_f_patch_size)} respectively. "
                        f"If the official z-image configs have been updated, ensure the current lightx2v's z-image model "
                        f"implementation matches the new configs then update this check."
                    )

                model_config["patch_size"] = z_image_patch_size[0]
                model_config["f_patch_size"] = z_image_f_patch_size[0]

            config.update(model_config)
        # load quantized config
        if config.get("dit_quantized_ckpt", None) is not None:
            config_path = os.path.join(config["dit_quantized_ckpt"], "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    model_config = json.load(f)
                config.update(model_config)

    if config["task"] in ["i2v", "s2v", "rs2v"]:
        if config["target_video_length"] % config["vae_stride"][0] != 1:
            logger.warning(f"`num_frames - 1` has to be divisible by {config['vae_stride'][0]}. Rounding to the nearest number.")
            config["target_video_length"] = config["target_video_length"] // config["vae_stride"][0] * config["vae_stride"][0] + 1

    # Load diffusers vae config
    if os.path.exists(os.path.join(config["model_path"], "vae", "config.json")):
        with open(os.path.join(config["model_path"], "vae", "config.json"), "r") as f:
            vae_config = json.load(f)
            if "temperal_downsample" in vae_config:
                config["vae_scale_factor"] = 2 ** len(vae_config["temperal_downsample"])
            elif "block_out_channels" in vae_config:
                config["vae_scale_factor"] = 2 ** (len(vae_config["block_out_channels"]) - 1)

    return config


def set_parallel_config(config):
    if config["parallel"]:
        tensor_p_size = config["parallel"].get("tensor_p_size", 1)

        if tensor_p_size > 1:
            # Tensor parallel only: 1D mesh
            assert tensor_p_size == dist.get_world_size(), f"tensor_p_size ({tensor_p_size}) must be equal to world_size ({dist.get_world_size()})"
            config["device_mesh"] = init_device_mesh(AI_DEVICE, (tensor_p_size,), mesh_dim_names=("tensor_p",))
            config["tensor_parallel"] = True
            config["seq_parallel"] = False
            config["cfg_parallel"] = False
        else:
            # Original 2D mesh for cfg_p and seq_p
            cfg_p_size = config["parallel"].get("cfg_p_size", 1)
            seq_p_size = config["parallel"].get("seq_p_size", 1)
            assert cfg_p_size * seq_p_size == dist.get_world_size(), f"cfg_p_size ({cfg_p_size}) * seq_p_size ({seq_p_size}) must be equal to world_size ({dist.get_world_size()})"
            config["device_mesh"] = init_device_mesh(AI_DEVICE, (cfg_p_size, seq_p_size), mesh_dim_names=("cfg_p", "seq_p"))
            config["tensor_parallel"] = False

            if config["parallel"] and config["parallel"].get("seq_p_size", False) and config["parallel"]["seq_p_size"] > 1:
                config["seq_parallel"] = True

            if config.get("enable_cfg", False) and config["parallel"] and config["parallel"].get("cfg_p_size", False) and config["parallel"]["cfg_p_size"] > 1:
                config["cfg_parallel"] = True

        # warmup dist
        _a = torch.zeros([1]).to(f"{AI_DEVICE}:{dist.get_rank()}")
        dist.all_reduce(_a)


def print_config(config):
    config_to_print = config.copy()
    if is_main_process():
        logger.info(f"config:\n{json.dumps(config_to_print, ensure_ascii=False, indent=4, default=str)}")
