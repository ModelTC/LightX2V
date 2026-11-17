import os

from lightx2v.models.runners.config import read_config
from lightx2v.models.runners.h3.config import VAE_CONFIG
from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    model_path = deployment_config["model_path"]
    subfolder = "transformer_diffusers" if os.path.isfile(os.path.join(model_path, "transformer_diffusers", "config.json")) else "transformer"
    transformer_path = deployment_config.get("dit_original_ckpt") or os.path.join(model_path, subfolder)
    config = {"patch_size": (1, 2, 2)} | read_config(os.path.join(transformer_path, "config.json"))
    for source, target in {
        "token_refiner_num_layers": "num_refiner_layers",
        "ffn_hidden_size": "ffn_dim",
        "latents_dim": "in_channels",
        "timestep_input_dim": "freq_dim",
        "time_embed_hidden_size": "time_embed_hidden_dim",
        "rope_inv_freq_len": "rope_freq_dim",
    }.items():
        if source in config:
            config[target] = config[source]
    config["dit_original_ckpt"] = transformer_path
    config["pdd_heads_path"] = os.path.join(model_path, "pdd_fused_heads.safetensors")
    config["pdd_config_path"] = os.path.join(model_path, "pdd_config.json")
    for component in ("text_encoder", "tokenizer", "processor"):
        config[f"{component}_path"] = os.path.join(model_path, component)
    config.update(infer_steps=6, num_frames=1, size=[2048, 2048], sample_guide_scale=1.0, vae_stride=[1, 16, 16])
    config.update(VAE_CONFIG)
    return merge_config(config, deployment_config)
