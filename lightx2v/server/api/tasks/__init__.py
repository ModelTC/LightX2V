from .common import router as common_router
from .image import router as image_router
from .video import router as video_router

__all__ = [
    "common_router",
    "video_router",
    "image_router",
]
