from lightx2v_train.utils.registry import build_model

from .flux2_klein import Flux2KleinModel
from .longcat_image import LongCatImageModel
from .qwen_image import QwenImageModel
from .wan_t2v import WanT2VModel
from .wan_ti2v_5b import WanTI2V5BModel

__all__ = ["build_model", "QwenImageModel", "LongCatImageModel", "Flux2KleinModel", "WanT2VModel", "WanTI2V5BModel"]
