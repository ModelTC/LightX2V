from lightx2v_train.utils.registry import build_inferencer

from .image import ImageInferencer
from .image_native import NativeImageInferencer
from .video import WanT2VARInferencer, WanT2VInferencer

__all__ = ["build_inferencer", "ImageInferencer", "NativeImageInferencer", "WanT2VInferencer", "WanT2VARInferencer"]
