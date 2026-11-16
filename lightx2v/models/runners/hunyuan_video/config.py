import os

from lightx2v.models.runners.config import load_vae_config, normalize_num_frames, read_config
from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    model_path = deployment_config["model_path"]
    transformer_path = os.path.join(model_path, "transformer", deployment_config["transformer_model_name"])
    config = {
        "vae_stride": (4, 8, 8),
        "patch_size": (1, 2, 2),
        "use_image_encoder": True,
        "transformer_model_path": transformer_path,
    }
    path = os.path.join(transformer_path, "config.json")
    if os.path.exists(path):
        config.update(read_config(path))
    config.update(load_vae_config(deployment_config))
    config = merge_config(config, deployment_config)
    normalize_num_frames(config)
    return config
