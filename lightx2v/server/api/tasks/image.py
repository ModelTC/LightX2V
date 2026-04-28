import asyncio
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from loguru import logger

from ...schema import ImageTaskRequest, TaskResponse
from ...task_manager import TaskStatus, task_manager
from ..deps import get_services, validate_url_async

router = APIRouter()


def _write_file_sync(file_path: Path, content: bytes) -> None:
    with open(file_path, "wb") as buffer:
        buffer.write(content)


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

        message.prefer_memory_result = False
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

        message.prefer_memory_result = True
        task_id = task_manager.create_task(message)
        message.task_id = task_id

        wait_task = asyncio.create_task(_wait_task_and_stream_result(task_id, timeout_seconds, poll_interval_seconds))
        disconnect_task = asyncio.create_task(_watch_client_disconnect(request, task_id))

        done, pending = await asyncio.wait({wait_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED)
        for pending_task in pending:
            pending_task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        if disconnect_task in done and disconnect_task.result():
            if not wait_task.done():
                wait_task.cancel()
                await asyncio.gather(wait_task, return_exceptions=True)
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
    image_file: UploadFile = File(None),
    prompt: str = Form(default=""),
    save_result_path: str = Form(default=""),
    use_prompt_enhancer: bool = Form(default=False),
    negative_prompt: str = Form(default=""),
    infer_steps: int = Form(default=5),
    seed: int = Form(default=42),
    aspect_ratio: str = Form(default="16:9"),
):
    services = get_services()
    assert services.file_service is not None, "File service is not initialized"

    async def save_file_async(file: UploadFile, target_dir: Path) -> str:
        if not file or not file.filename:
            return ""

        file_extension = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = target_dir / unique_filename

        content = await file.read()
        await asyncio.to_thread(_write_file_sync, file_path, content)

        return str(file_path)

    image_path = ""
    if image_file and image_file.filename:
        image_path = await save_file_async(image_file, services.file_service.input_image_dir)

    message = ImageTaskRequest(
        prompt=prompt,
        use_prompt_enhancer=use_prompt_enhancer,
        negative_prompt=negative_prompt,
        image_path=image_path,
        save_result_path=save_result_path,
        infer_steps=infer_steps,
        seed=seed,
        aspect_ratio=aspect_ratio,
    )

    try:
        message.prefer_memory_result = False
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
        logger.error(f"Failed to create image form task: {e}")
        raise HTTPException(status_code=500, detail=str(e))
