from lightx2v_train.utils.registry import build_model

from .flux2_klein import Flux2KleinModel
from .longcat_image import LongCatImageModel
from .qwen_image import QwenImageModel
from .qwen_image_edit import QwenImageEditModel
from .wan_t2v import WanT2VModel

__all__ = ["build_model", "QwenImageModel", "QwenImageEditModel", "LongCatImageModel", "Flux2KleinModel", "WanT2VModel"]
