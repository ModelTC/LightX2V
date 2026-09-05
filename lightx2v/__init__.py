__version__ = "0.1.0"
__author__ = "LightX2V Contributors"
__license__ = "Apache 2.0"

import os

import lightx2v_platform.set_ai_device
from lightx2v import common, models, utils

if os.getenv("LIGHTX2V_SKIP_PIPELINE_IMPORT", "0").lower() in ("1", "true", "yes", "on"):
    LightX2VPipeline = None
else:
    from lightx2v.pipeline import LightX2VPipeline

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "models",
    "common",
    "utils",
    "LightX2VPipeline",
]
