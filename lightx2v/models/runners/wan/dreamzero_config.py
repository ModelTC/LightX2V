import os

from lightx2v.models.runners.config import load_vae_config, normalize_num_frames, read_config
from lightx2v.models.runners.wan.config import WAN_DEFAULTS
from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    model_path = deployment_config["model_path"]
    model_config = dict(WAN_DEFAULTS)
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        checkpoint_config = read_config(config_path)
        model_config.update(checkpoint_config)
        action_head_config = checkpoint_config.get("action_head_cfg", {}).get("config", {})
        diffusion_model_config = action_head_config.get("diffusion_model_cfg", {})
        model_config.update(action_head_config)
        model_config.update(diffusion_model_config)

    model_config.update(load_vae_config(deployment_config))
    config = merge_config(model_config, deployment_config)
    if "out_dim" in config:
        config["num_channels_latents"] = config["out_dim"]
    normalize_num_frames(config)
    return config
