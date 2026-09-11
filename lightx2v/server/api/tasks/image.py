import asyncio
import time

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from loguru import logger

from ...schema import ImageTaskRequest, TaskResponse
from ...task_manager import TaskStatus, task_manager
from ..deps import get_services, validate_url_async
from .common import parse_form_request

router = APIRouter()


async def _wait_task_and_stream_result(task_id: str, timeout_seconds: int, poll_interval_seconds: float):
    start_time = time.monotonic()
    while True:
        task_status = task_manager.get_task_status(task_id)
        if not task_status:
            raise HTTPException(status_code=500, detail=f"Task status not found: {task_id}")

        status = task_status.get("status")
        if status == TaskStatus.COMPLETED.value:
            result_png = task_manager.get_task_result_png(task_id)
            if result_png:
                return result_png
            raise HTTPException(status_code=500, detail=f"Task completed but no in-memory image found: {task_id}")

        if status == TaskStatus.FAILED.value:
            error_type = task_status.get("error_type", "")
            error_detail = task_status.get("error", "Task failed")
            if error_type == "ValueError":
                raise HTTPException(status_code=413, detail=error_detail)
            raise HTTPException(status_code=500, detail=error_detail)

        if status == TaskStatus.CANCELLED.value:
            raise HTTPException(status_code=409, detail=task_status.get("error", "Task cancelled"))

        if (time.monotonic() - start_time) > timeout_seconds:
            task_manager.cancel_task(task_id)
            raise HTTPException(status_code=504, detail=f"Task {task_id} timed out after {timeout_seconds} seconds")

        await asyncio.sleep(poll_interval_seconds)


def _build_png_response(result_png: bytes) -> Response:
    return Response(
        content=result_png,
        media_type="image/png",
        headers={"Content-Disposition": 'inline; filename="result.png"'},
    )


async def _upload_sync_result_if_needed(message: ImageTaskRequest, result_png: bytes):
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


async def _watch_client_disconnect(request: Request, task_id: str, poll_interval_seconds: float = 0.2) -> bool:
    while True:
        if await request.is_disconnected():
            task_manager.cancel_task(task_id)
            logger.info(f"Client disconnected, task {task_id} cancelled")
            return True
        await asyncio.sleep(poll_interval_seconds)


@router.post("/", response_model=TaskResponse)
async def create_image_task(message: ImageTaskRequest):
    try:
        if hasattr(message, "image_path") and message.image_path and message.image_path.startswith("http"):
            if not await validate_url_async(message.image_path):
                raise HTTPException(status_code=400, detail=f"Image URL is not accessible: {message.image_path}")
        if hasattr(message, "image_mask_path") and message.image_mask_path and message.image_mask_path.startswith("http"):
            if not await validate_url_async(message.image_mask_path):
                raise HTTPException(status_code=400, detail=f"Image mask URL is not accessible: {message.image_mask_path}")

        message._prefer_memory_result = False
        task_id = task_manager.create_task(message)
        message.task_id = task_id

        return TaskResponse(
            task_id=task_id,
            task_status="pending",
            save_result_path=message.save_result_path,
        )
    except RuntimeError as e:
        if getattr(e, "original_error_type", "") == "ValueError":
            raise HTTPException(status_code=413, detail=str(e))
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create image task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync")
async def create_image_task_sync(
    request: Request,
    message: ImageTaskRequest,
    timeout_seconds: int = 600,
    poll_interval_seconds: float = 0.5,
):
    if timeout_seconds <= 0:
        raise HTTPException(status_code=400, detail="timeout_seconds must be > 0")
    if poll_interval_seconds <= 0:
        raise HTTPException(status_code=400, detail="poll_interval_seconds must be > 0")

    task_id = None
    try:
        if hasattr(message, "image_path") and message.image_path and message.image_path.startswith("http"):
            if not await validate_url_async(message.image_path):
                raise HTTPException(status_code=400, detail=f"Image URL is not accessible: {message.image_path}")
        if hasattr(message, "image_mask_path") and message.image_mask_path and message.image_mask_path.startswith("http"):
            if not await validate_url_async(message.image_mask_path):
                raise HTTPException(status_code=400, detail=f"Image mask URL is not accessible: {message.image_mask_path}")
        if hasattr(message, "presigned_url") and message.presigned_url:
            if not message.presigned_url.startswith(("http://", "https://")):
                raise HTTPException(status_code=400, detail=f"Invalid presigned_url: {message.presigned_url}")

        message._prefer_memory_result = True
        task_id = task_manager.create_task(message)
        message.task_id = task_id

        wait_task = asyncio.create_task(_wait_task_and_stream_result(task_id, timeout_seconds, poll_interval_seconds))
        disconnect_task = asyncio.create_task(_watch_client_disconnect(request, task_id))

        done, pending = await asyncio.wait({wait_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED)
        for pending_task in pending:
            pending_task.cancel()

        if disconnect_task in done:
            if not wait_task.done():
                wait_task.cancel()
            raise HTTPException(status_code=499, detail=f"Client disconnected, task {task_id} cancelled")

        result_png = wait_task.result()
        upload_result = await _upload_sync_result_if_needed(message, result_png)
        if upload_result is not None:
            return upload_result
        return _build_png_response(result_png)

    except asyncio.CancelledError:
        if task_id:
            task_manager.cancel_task(task_id)
        raise

    except RuntimeError as e:
        if getattr(e, "original_error_type", "") == "ValueError":
            raise HTTPException(status_code=413, detail=str(e))
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to run sync image task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/form", response_model=TaskResponse)
async def create_image_task_form(
    request: Request,
    task: str | None = Form(default=None),
    image_file: UploadFile = File(None),
    prompt: str = Form(default=""),
    save_result_path: str = Form(default=""),
    negative_prompt: str = Form(default=""),
    seed: int | None = Form(default=None),
    aspect_ratio: str | None = Form(default=None),
):
    services = get_services()
    assert services.file_service is not None, "File service is not initialized"

    image_path = ""
    if image_file and image_file.filename:
        content = await image_file.read()
        image_path = str(await asyncio.to_thread(services.file_service.save_uploaded_file, content, image_file.filename))

    request_data = {"seed": seed} if seed is not None else {}
    if image_path:
        request_data["image_path"] = image_path
    # FastAPI replaces empty form strings with defaults; preserve submitted text.
    form = await request.form()
    form_data = {key: value for key, value in form.items() if key != "image_file"}
    message = parse_form_request(ImageTaskRequest, form_data | request_data)

    return await create_image_task(message)
