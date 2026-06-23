from lightx2v_train.utils.registry import build_model

from .flux2_klein import Flux2KleinModel
from .flux2_dev import Flux2DevModel
from .longcat_image import LongCatImageModel
from .qwen_image import QwenImageModel
from .wan_t2v import WanT2VModel

__all__ = ["build_model", "QwenImageModel", "LongCatImageModel", "Flux2KleinModel", "Flux2DevModel", "WanT2VModel"]
