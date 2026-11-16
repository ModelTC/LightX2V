from fastapi import APIRouter, Request

from ...schema import TaskResponse
from ..deps import get_services
from ..request_parsing import form_request_openapi, parse_upload_form
from .submission import submit_task


def create_router(request_type):
    router = APIRouter()
    uploads = {"image_file": "image_path", "last_frame_file": "last_frame_path", "audio_file": "audio_path"}

    @router.post("/video/", response_model=TaskResponse)
    async def create_video_task(message: request_type):
        return await submit_task(message, required_output="video")

    @router.post("/video/form", response_model=TaskResponse, openapi_extra=form_request_openapi(request_type, uploads=uploads))
    async def create_video_task_form(request: Request):
        async with request.form() as form:
            message = await parse_upload_form(request_type, form, get_services().file_service, uploads)
        return await submit_task(message, required_output="video")

    return router
