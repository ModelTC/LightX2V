__version__ = "0.5.0"
__author__ = "LightX2V Contributors"
__license__ = "Apache 2.0"

import os

# Must run before lightx2v_platform: platform init can initialize the CUDA/HIP caching
# allocator (e.g. importing aiter on amd_rocm), after which PYTORCH_CUDA_ALLOC_CONF is ignored.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import lightx2v_platform.set_ai_device
from lightx2v import common, models, utils
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
