from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from lightx2v.utils.image_output import image_format_for_bytes

from ..deps import get_services
from .submission import run_image_task


def _build_image_response(result_png: bytes) -> Response:
    fmt = image_format_for_bytes(result_png)
    return Response(
        content=result_png,
        media_type=f"image/{fmt}",
        headers={"Content-Disposition": f'inline; filename="result.{fmt}"'},
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
            content_type="image/" + image_format_for_bytes(result_png),
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

    @router.post("/sync", responses={200: {"content": {**{f"image/{fmt}": {"schema": {"type": "string", "format": "binary"}} for fmt in ("png", "jpeg", "webp")}}}})
    async def create_image_task_sync(
        request: Request,
        message: request_type,
        timeout_seconds: int = 600,
        poll_interval_seconds: float = 0.5,
    ):
        """Generate an image synchronously, returning PNG/JPEG/WebP or a presigned upload receipt."""
        if message.presigned_url and not message.presigned_url.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail=f"Invalid presigned_url: {message.presigned_url}")
        png, _ = await run_image_task(request, message, timeout_seconds=timeout_seconds, poll_interval_seconds=poll_interval_seconds)
        uploaded = await _upload_sync_result_if_needed(message, png)
        return uploaded if uploaded is not None else _build_image_response(png)

    return router
