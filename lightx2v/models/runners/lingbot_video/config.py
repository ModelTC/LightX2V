from lightx2v.models.runners.config import load_vae_config, normalize_num_frames, read_checkpoint_config
from lightx2v.utils.request_defaults import set_input_value
from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    config = merge_config(
        {"vae_stride": (4, 8, 8), "patch_size": (1, 2, 2)} | read_checkpoint_config(deployment_config["model_path"], ("transformer",)) | load_vae_config(deployment_config), deployment_config
    )
    if "num_frames" not in config and "num_frames" not in config["request_defaults"]:
        set_input_value(config, "num_frames", 1)
    config.setdefault("fps", 24)
    config["vae_scale_factor_spatial"] = int(config.get("vae_scale_factor_spatial", 8))
    config["vae_scale_factor_temporal"] = int(config.get("vae_scale_factor_temporal", 4))
    config["vae_scale_factor"] = config["vae_scale_factor_spatial"]
    normalize_num_frames(config)
    return config
