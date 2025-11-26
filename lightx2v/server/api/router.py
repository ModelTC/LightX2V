from fastapi import APIRouter

from .files import router as files_router
from .service_routes import router as service_router
from .tasks import common_router, image_router, video_router


def create_api_router() -> APIRouter:
    api_router = APIRouter()

    tasks_router = APIRouter(prefix="/v1/tasks", tags=["tasks"])
    tasks_router.include_router(common_router)
    tasks_router.include_router(video_router, prefix="/video", tags=["video"])
    tasks_router.include_router(image_router, prefix="/image", tags=["image"])

    # backward compatibility : POST /v1/tasks default to video task
    from .tasks.video import create_video_task

    tasks_router.post("/", response_model_exclude_unset=True, deprecated=True)(create_video_task)

    api_router.include_router(tasks_router)
    api_router.include_router(files_router, prefix="/v1/files", tags=["files"])
    api_router.include_router(service_router, prefix="/v1/service", tags=["service"])

    return api_router
