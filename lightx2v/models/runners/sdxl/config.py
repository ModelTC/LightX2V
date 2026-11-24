"""Model-specific deployment defaults; no inference runtime is imported."""

from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    return merge_config({"enable_cfg": True, "warmup": False, "parallel": False, "request_defaults": {"seed": 42, "size": [1024, 1024]}, "infer_steps": 30, "cfg_scale": 5.0}, deployment_config)
