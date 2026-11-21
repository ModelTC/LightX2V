import os

from lightx2v import common, models, utils

os.environ.setdefault("PROFILING_DEBUG_LEVEL", "2")
os.environ.setdefault("DTYPE", "BF16")
os.environ.setdefault("SENSITIVE_LAYER_DTYPE", "None")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def __getattr__(name):
    if name == "LightX2VPipeline":
        from lightx2v.pipeline import LightX2VPipeline

        globals()[name] = LightX2VPipeline
        return LightX2VPipeline
    raise AttributeError(name)


__all__ = [
    "models",
    "common",
    "utils",
    "LightX2VPipeline",
]
