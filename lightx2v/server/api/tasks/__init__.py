from .common import router as common_router
from .image import router as image_router
from .sensenova_vision import router as sensenova_vision_router
from .video import router as video_router

__all__ = [
    "common_router",
    "video_router",
    "image_router",
    "sensenova_vision_router",
]
