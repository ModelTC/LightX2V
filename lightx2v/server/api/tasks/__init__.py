from .common import router as common_router
from .image import create_router as create_image_router
from .video import create_router as create_video_router

__all__ = [
    "common_router",
    "create_video_router",
    "create_image_router",
]
