import os

from lightx2v.models.runners.config import CHECKPOINT_SUBFOLDERS, load_quantized_config, load_vae_config, read_config
from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    config = {}
    model_path = deployment_config["model_path"]
    for subfolder in CHECKPOINT_SUBFOLDERS:
        path = os.path.join(model_path, subfolder, "config.json")
        if not os.path.exists(path):
            continue
        config.update(read_config(path))
        if subfolder == "transformer":
            patch_size = config.pop("all_patch_size", [2])
            f_patch_size = config.pop("all_f_patch_size", [1])
            if len(patch_size) != 1 or len(f_patch_size) != 1:
                raise ValueError(f"Z-Image requires one spatial and temporal patch size; got all_patch_size={patch_size!r}, all_f_patch_size={f_patch_size!r}.")
            config["patch_size"] = patch_size[0]
            config["f_patch_size"] = f_patch_size[0]
        break
    config.update(load_quantized_config(deployment_config))
    config.update(load_vae_config(deployment_config))
    return merge_config(config, deployment_config)
