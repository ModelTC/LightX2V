import os

from lightx2v.models.runners.config import load_quantized_config, read_checkpoint_config, read_config
from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    model_path = deployment_config["model_path"]
    config = read_checkpoint_config(model_path) | load_quantized_config(deployment_config)
    path = os.path.join(model_path, "vae", "config.json")
    if os.path.exists(path):
        config["vae_scale_factor"] = 2 ** len(read_config(path)["block_out_channels"])
    return merge_config(config, deployment_config)
