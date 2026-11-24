"""Model-specific deployment defaults; no inference runtime is imported."""

from lightx2v.utils.set_config import merge_config


def load_config(deployment_config):
    return merge_config(
        {"enable_cfg": False, "warmup": False, "parallel": False, "request_defaults": {"seed": 42}, "bake_resolution": 1024, "foreground_ratio": 0.85, "deterministic_algorithms": True},
        deployment_config,
    )
