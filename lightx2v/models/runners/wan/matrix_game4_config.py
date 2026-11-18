import os

from lightx2v.models.runners.config import load_vae_config, normalize_num_frames, read_config
from lightx2v.models.runners.wan.config import WAN_DEFAULTS
from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    model_path = deployment_config["model_path"]
    model_config = dict(WAN_DEFAULTS)
    release = read_config(os.path.join(model_path, "release-config.json"))
    variant = deployment_config.get("model_variant", "castle")
    if variant not in release["scenes"]:
        raise ValueError(f"Unknown Matrix-Game-4 model_variant {variant!r}; choose from {tuple(release['scenes'])}")
    model_config.update(release["architecture"]["dit_config"])
    model_config.update(model_variant=variant, num_channels_latents=model_config["out_dim"], vae_stride=[4, 16, 16])
    component_files = {
        "dit_original_ckpt": f"models/{variant}/model.safetensors",
        "t5_original_ckpt": "base/models_t5_umt5-xxl-enc-bf16.pth",
        "t5_tokenizer_path": "base/google/umt5-xxl",
        "vae_path": "base/Wan2.2_VAE.pth",
        "tae_path": "base/taew2_2.pth",
    }
    for key, filename in component_files.items():
        model_config[key] = deployment_config.get(key, os.path.join(model_path, filename))

    model_config.update(load_vae_config(deployment_config))
    config = merge_config(model_config, deployment_config)
    normalize_num_frames(config)
    return config
