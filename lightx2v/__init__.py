__version__ = "0.1.0"
__author__ = "LightX2V Contributors"
__license__ = "Apache 2.0"

import lightx2v_platform.set_ai_device
from lightx2v import common, models, utils


def __getattr__(name):
    # Importing a model-specific runner (including ROS and RoboDojo adapters)
    # should not eagerly require every optional dependency in pipeline.py.
    if name == "LightX2VPipeline":
        from lightx2v.pipeline import LightX2VPipeline

        return LightX2VPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "models",
    "common",
    "utils",
    "LightX2VPipeline",
]
