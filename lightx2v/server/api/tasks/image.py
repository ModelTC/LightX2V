from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from ...schema import TaskResponse
from ..deps import get_services
from ..request_parsing import form_request_openapi, parse_upload_form
from .submission import run_image_task, submit_task


def _build_png_response(result_png: bytes) -> Response:
    return Response(
        content=result_png,
        media_type="image/png",
        headers={"Content-Disposition": 'inline; filename="result.png"'},
    )


async def _upload_sync_result_if_needed(message, result_png: bytes):
    presigned_url = (getattr(message, "presigned_url", "") or "").strip()
    if not presigned_url:
        return None

    services = get_services()
    assert services.file_service is not None, "File service is not initialized"

    try:
        await services.file_service.upload_to_presigned_url(
            presigned_url=presigned_url,
            file_content=result_png,
            content_type="image/png",
        )
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"Failed to upload sync result to presigned URL: {str(e)}")

    return {
        "task_id": message.task_id,
        "task_status": "completed",
        "uploaded_to_presigned_url": True,
        "presigned_url": presigned_url,
    }


def create_router(request_type):
    router = APIRouter()
    uploads = {"image_file": "image_path"}

    @router.post("/", response_model=TaskResponse)
    async def create_image_task(message: request_type):
        return await submit_task(message, required_output="image")

    @router.post("/sync")
    async def create_image_task_sync(
        request: Request,
        message: request_type,
        timeout_seconds: int = 600,
        poll_interval_seconds: float = 0.5,
    ):
        if message.presigned_url and not message.presigned_url.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail=f"Invalid presigned_url: {message.presigned_url}")
        png, _ = await run_image_task(request, message, timeout_seconds=timeout_seconds, poll_interval_seconds=poll_interval_seconds)
        uploaded = await _upload_sync_result_if_needed(message, png)
        return uploaded if uploaded is not None else _build_png_response(png)

    @router.post("/form", response_model=TaskResponse, openapi_extra=form_request_openapi(request_type, uploads=uploads))
    async def create_image_task_form(request: Request):
        async with request.form() as form:
            message = await parse_upload_form(request_type, form, get_services().file_service, uploads)
        return await submit_task(message, required_output="image")

    return router
