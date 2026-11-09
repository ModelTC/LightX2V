from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from ...schema import TaskResponse
from ...task_manager import task_manager
from ..deps import get_services, validate_image_urls
from ..request_parsing import form_request_openapi, parse_upload_form


def create_router(request_type):
    router = APIRouter()
    uploads = {"image_file": "image_path", "last_frame_file": "last_frame_path", "audio_file": "audio_path"}

    @router.post("/", response_model=TaskResponse, deprecated=True)
    @router.post("/video/", response_model=TaskResponse)
    async def create_video_task(message: request_type):
        try:
            await validate_image_urls(message)

            task_id = task_manager.create_task(message)
            message.task_id = task_id

            return TaskResponse(
                task_id=task_id,
                task_status="pending",
                save_result_path=message.save_result_path,
            )
        except HTTPException:
            raise
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to create video task: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/video/form",
        response_model=TaskResponse,
        openapi_extra=form_request_openapi(request_type, uploads=uploads),
    )
    async def create_video_task_form(request: Request):
        services = get_services()
        assert services.file_service is not None, "File service is not initialized"

        async with request.form() as form:
            message = await parse_upload_form(request_type, form, services.file_service, uploads)
        return await create_video_task(message)

    return router
