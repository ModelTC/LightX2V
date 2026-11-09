from fastapi import APIRouter

from ..schema import ImageTaskRequest, VideoTaskRequest, build_task_request
from .files import router as files_router
from .openai_images import create_router as create_openai_images_router
from .service_routes import router as service_router
from .tasks import common_router, create_image_router, create_video_router


def create_api_router(config) -> APIRouter:
    api_router = APIRouter()
    image_request = build_task_request(config, ImageTaskRequest, "image")
    video_request = build_task_request(config, VideoTaskRequest, "video")
    if image_request is None and video_request is None:
        raise ValueError(f"{config['model_cls']} has no tasks supported by the image or video APIs")

    tasks_router = APIRouter(prefix="/v1/tasks", tags=["tasks"])
    tasks_router.include_router(common_router)
    if video_request is not None:
        tasks_router.include_router(create_video_router(video_request), tags=["video"])
    if image_request is not None:
        tasks_router.include_router(create_image_router(image_request), prefix="/image", tags=["image"])

    api_router.include_router(tasks_router)
    if image_request is not None:
        api_router.include_router(create_openai_images_router(config, image_request), prefix="/v1/images", tags=["openai-images"])
    api_router.include_router(files_router, prefix="/v1/files", tags=["files"])
    api_router.include_router(service_router, prefix="/v1/service", tags=["service"])

    return api_router
