from lightx2v.models.runners.config import load_vae_config, normalize_num_frames, read_checkpoint_config
from lightx2v.models.runners.wan.config import WAN_DEFAULTS
from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    config = merge_config(
        WAN_DEFAULTS
        | {"text_dim": 4096, "window_size": (-1, -1), "qk_norm": True, "cross_attn_norm": True}
        | read_checkpoint_config(deployment_config["model_path"], ("",))
        | load_vae_config(deployment_config),
        deployment_config,
    )
    normalize_num_frames(config)
    return config
