import os

from lightx2v.models.runners.config import load_vae_config
from lightx2v.models.runners.wan.config import WAN_DEFAULTS
from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    model_path = deployment_config["model_path"]
    model_config = dict(WAN_DEFAULTS)
    component_files = {
        "dit_original_ckpt": "tbdub_student.safetensors",
        "vae_path": "Wan2.2_VAE.safetensors",
        "audio_encoder_path": "hubert-large-ll60k",
        "prompt_embedding_path": "null_prompt_emb.pt",
    }
    for key, filename in component_files.items():
        model_config[key] = deployment_config.get(key, os.path.join(model_path, filename))

    model_config.update(load_vae_config(deployment_config))
    return merge_config(model_config, deployment_config)
