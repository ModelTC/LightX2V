import os

from lightx2v.models.runners.config import load_vae_config
from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    model_path = deployment_config["model_path"]
    candidate = os.path.join(model_path, deployment_config.get("subfolder", "HY-WorldMirror-2.0"))
    transformer_path = candidate if os.path.isdir(candidate) else model_path
    return merge_config({"transformer_model_path": transformer_path} | load_vae_config(deployment_config), deployment_config)
