from lightx2v.models.runners.config import load_checkpoint_config, normalize_num_frames
from lightx2v.utils.set_config import merge_config

WAN_DEFAULTS = {
    "vae_stride": (4, 8, 8),
    "patch_size": (1, 2, 2),
    "use_image_encoder": True,
}


def load_config(deployment_config):
    config = merge_config(WAN_DEFAULTS | load_checkpoint_config(deployment_config), deployment_config)
    normalize_num_frames(config)
    return config
