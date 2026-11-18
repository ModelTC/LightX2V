import os

from lightx2v.models.runners.config import normalize_num_frames, read_checkpoint_config, read_config
from lightx2v.utils.request_defaults import set_input_value
from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    model_path = deployment_config["model_path"]
    config = {"vae_stride": (4, 8, 8)} | read_checkpoint_config(model_path, ("transformer",))
    vae_path = os.path.join(model_path, "vae", "config.json")
    if os.path.exists(vae_path):
        vae = read_config(vae_path)
        config["vae_scale_factor_spatial"] = int(vae.get("scale_factor_spatial", 16))
        config["vae_scale_factor_temporal"] = int(vae.get("scale_factor_temporal", 4))
        config["vae_scale_factor"] = config["vae_scale_factor_spatial"]
    sound_path = os.path.join(model_path, "sound_tokenizer", "config.json")
    if os.path.exists(sound_path):
        sound = read_config(sound_path)
        config["sound_sampling_rate"] = int(sound.get("sampling_rate", 48000))
        config["sound_hop_size"] = int(sound.get("hop_size", 1920))
    config = merge_config(config, deployment_config)
    if "num_frames" not in config and "num_frames" not in config["request_defaults"]:
        set_input_value(config, "num_frames", 1)
    config.setdefault("fps", config.get("base_fps", 24))
    if config.get("model_variant") not in ("omni", "t2i", "i2v", "policy"):
        raise ValueError("Cosmos3 requires model_variant='omni', 't2i', 'i2v', or 'policy' in the startup config.")
    normalize_num_frames(config)
    return config
