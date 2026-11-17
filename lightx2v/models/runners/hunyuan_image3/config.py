from lightx2v.models.networks.hunyuan_image3.config import normalize_hunyuan_image3_config
from lightx2v.models.runners.config import load_checkpoint_config
from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    config = merge_config(load_checkpoint_config(deployment_config), deployment_config)
    normalize_hunyuan_image3_config(config)
    return config
