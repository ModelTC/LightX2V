__version__ = "0.1.0"
__author__ = "LightX2V Contributors"
__license__ = "Apache 2.0"

# AMD ROCm optimizations (must be before other imports)
from lightx2v_platform.base.amd_rocm import init_amd_rocm
init_amd_rocm()

import lightx2v_platform.set_ai_device
from lightx2v import common, deploy, models, utils
from lightx2v.pipeline import LightX2VPipeline

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "models",
    "common",
    "deploy",
    "utils",
    "LightX2VPipeline",
]
