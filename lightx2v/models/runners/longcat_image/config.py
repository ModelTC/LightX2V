import os

from lightx2v.models.runners.config import load_vae_config, read_config
from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    config = {}
    for subfolder in ("", "transformer"):
        path = os.path.join(deployment_config["model_path"], subfolder, "config.json")
        if os.path.exists(path):
            config.update(read_config(path))
    config.update(load_vae_config(deployment_config))
    return merge_config(config, deployment_config)
