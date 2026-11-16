from fastapi import APIRouter, Request

from ...schema import TaskResponse
from ..deps import get_services
from ..request_parsing import form_request_openapi, parse_upload_form
from .submission import submit_task

UPLOADS = {
    "image_file": "image_path",
    "last_frame_file": "last_frame_path",
    "audio_file": "audio_path",
    "video_file": "video_path",
    "state_file": "state_path",
    "action_file": "action_path",
}


def create_router(request_type):
    router = APIRouter()

    @router.post("/", response_model=TaskResponse)
    async def create_task(message: request_type):
        return await submit_task(message)

    @router.post("/form", response_model=TaskResponse, openapi_extra=form_request_openapi(request_type, uploads=UPLOADS))
    async def create_task_form(request: Request):
        async with request.form() as form:
            message = await parse_upload_form(request_type, form, get_services().file_service, UPLOADS)
        return await submit_task(message)

    return router
