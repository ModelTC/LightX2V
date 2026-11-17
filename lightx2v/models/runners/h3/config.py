from lightx2v.utils.set_config import merge_config

VAE_CONFIG = {"vae_spatial_scale_factor": 16, "vae_scale_factor": 16}


def prepare_h3_config(config):
    if config.get("fps", 24) != 24:
        raise ValueError(f"MiniMax-H3 requires fps=24 to keep video and audio timing aligned; got {config['fps']!r}")
    config.setdefault("fps", 24)
    if config.get("dit_quantized_ckpt"):
        config["dit_quantized"] = True
        config["dit_quant_scheme"] = {"fp8": "fp8-q8f", "int8": "int8-q8f"}.get(config.get("dit_quant_scheme"), config.get("dit_quant_scheme", "Default"))


def load_config(deployment_config):
    config = merge_config(VAE_CONFIG, deployment_config)
    prepare_h3_config(config)
    return config
