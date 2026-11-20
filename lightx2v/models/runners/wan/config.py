from lightx2v.models.runners.config import load_checkpoint_config, normalize_num_frames
from lightx2v.utils.set_config import merge_config

WAN_DEFAULTS = {
    "vae_stride": (4, 8, 8),
    "patch_size": (1, 2, 2),
    "use_image_encoder": True,
}


def load_config(deployment_config):
    config = merge_config(WAN_DEFAULTS | load_checkpoint_config(deployment_config), deployment_config)
    normalize_num_frames(config)
    return config


def load_animate2_config(deployment_config):
    config = load_config(deployment_config)
    if config.get("lora_configs") and config.get("lora_mode", "merge") == "merge" and int((config.get("parallel") or {}).get("tensor_p_size", 1)) > 1:
        raise ValueError("Wan-Animate-2 cannot merge LoRA into tensor-parallel weights. Use a checkpoint with the LoRA merged offline, or set parallel.tensor_p_size=1.")
    return config
