import asyncio
import base64
import re
import time
import uuid
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from loguru import logger
from pydantic import BaseModel, Field

from ..schema import ImageTaskRequest
from ..task_manager import TaskStatus, task_manager
from .deps import get_services

router = APIRouter()

_SIZE_PATTERN = re.compile(r"^\s*(\d+)\s*x\s*(\d+)\s*$", re.IGNORECASE)


class OpenAIImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    model: Optional[str] = Field(default=None, description="Ignored for compatibility")
    n: int = Field(default=1, description="Number of images, currently only supports 1")
    size: Optional[str] = Field(default=None, description="Image size, e.g. 1024x1024")
    response_format: Literal["b64_json"] = Field(default="b64_json")
    user: Optional[str] = Field(default=None, description="Ignored for compatibility")
    seed: Optional[int] = Field(default=None, description="Optional random seed")


class OpenAIImageResponse(BaseModel):
    created: int
    data: list[dict[str, str]]


def _write_file_sync(file_path: Path, content: bytes) -> None:
    with open(file_path, "wb") as buffer:
        buffer.write(content)


def _shape_from_size(size: str) -> tuple[int, int]:
    match = _SIZE_PATTERN.match(size)
    if not match:
        raise ValueError("size must be in WxH format, e.g. 1024x1024")
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError("size width and height must be positive")
    return width, height


async def _wait_task_result_png(task_id: str, timeout_seconds: int, poll_interval_seconds: float) -> bytes:
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
            raise HTTPException(status_code=500, detail=task_status.get("error", "Task failed"))

        if status == TaskStatus.CANCELLED.value:
            raise HTTPException(status_code=409, detail=task_status.get("error", "Task cancelled"))

        if (time.monotonic() - start_time) > timeout_seconds:
            task_manager.cancel_task(task_id)
            raise HTTPException(status_code=504, detail=f"Task {task_id} timed out after {timeout_seconds} seconds")

        await asyncio.sleep(poll_interval_seconds)


async def _watch_client_disconnect(request: Request, task_id: str, poll_interval_seconds: float = 0.2) -> bool:
    while True:
        if await request.is_disconnected():
            task_manager.cancel_task(task_id)
            logger.info(f"Client disconnected, task {task_id} cancelled")
            return True
        await asyncio.sleep(poll_interval_seconds)


async def _run_sync_image_task(request: Request, message: ImageTaskRequest) -> bytes:
    task_id = None
    timeout_seconds = 600
    poll_interval_seconds = 0.5

    try:
        message.prefer_memory_result = True
        task_id = task_manager.create_task(message)
        message.task_id = task_id

        wait_task = asyncio.create_task(_wait_task_result_png(task_id, timeout_seconds, poll_interval_seconds))
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

        return wait_task.result()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except asyncio.CancelledError:
        if task_id:
            task_manager.cancel_task(task_id)
        raise
    except Exception as e:
        logger.error(f"Failed to run OpenAI-compatible image task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _build_url_response(request: Request, task_id: str, image_bytes: bytes) -> str:
    services = get_services()
    assert services.file_service is not None, "File service is not initialized"

    file_name = f"{task_id}.png"
    output_path = services.file_service.output_video_dir / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_file_sync(output_path, image_bytes)

    base = str(request.base_url).rstrip("/")
    return f"{base}/v1/files/download/{file_name}"


def _build_openai_response(request: Request, task_id: str, image_bytes: bytes, response_format: Literal["url", "b64_json"]):
    if response_format == "b64_json":
        return OpenAIImageResponse(created=int(time.time()), data=[{"b64_json": base64.b64encode(image_bytes).decode("utf-8")}])

    return OpenAIImageResponse(created=int(time.time()), data=[{"url": _build_url_response(request, task_id, image_bytes)}])


def _build_image_task_request(
    prompt: str,
    *,
    negative_prompt: str = "",
    seed: Optional[int] = None,
    target_shape: Optional[list[int]] = None,
    image_path: str = "",
    image_mask_path: str = "",
) -> ImageTaskRequest:
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image_path": image_path,
        "image_mask_path": image_mask_path,
    }
    if target_shape:
        payload["target_shape"] = target_shape
    if seed is not None:
        payload["seed"] = seed
    return ImageTaskRequest(**payload)


@router.post("/generations", response_model=OpenAIImageResponse)
async def create_openai_image_generation(request: Request, body: OpenAIImageGenerationRequest):
    if body.n != 1:
        raise HTTPException(status_code=400, detail="Only n=1 is currently supported")
    if not body.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    target_shape = None
    if body.size:
        try:
            width, height = _shape_from_size(body.size)
            target_shape = [height, width]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    message = _build_image_task_request(
        prompt=body.prompt,
        seed=body.seed,
        target_shape=target_shape,
    )

    result_png = await _run_sync_image_task(request, message)
    return _build_openai_response(request, message.task_id, result_png, body.response_format)


async def _save_upload_file(file: UploadFile, target_dir: Path) -> str:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename")

    file_extension = Path(file.filename).suffix or ".png"
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = target_dir / unique_filename

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"Uploaded file is empty: {file.filename}")
    await asyncio.to_thread(_write_file_sync, file_path, content)
    return str(file_path)


@router.post("/edits", response_model=OpenAIImageResponse)
async def create_openai_image_edit(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    mask: UploadFile | None = File(default=None),
    model: str | None = Form(default=None),
    n: int = Form(default=1),
    size: str | None = Form(default=None),
    response_format: Literal["url", "b64_json"] = Form(default="url"),
    user: str | None = Form(default=None),
    negative_prompt: str = Form(default=""),
    seed: int | None = Form(default=None),
):
    _ = model, user
    if n != 1:
        raise HTTPException(status_code=400, detail="Only n=1 is currently supported")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    services = get_services()
    assert services.file_service is not None, "File service is not initialized"

    target_shape = None
    if size:
        try:
            width, height = _shape_from_size(size)
            target_shape = [height, width]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    image_path = await _save_upload_file(image, services.file_service.input_image_dir)
    image_mask_path = ""
    if mask is not None:
        image_mask_path = await _save_upload_file(mask, services.file_service.input_image_dir)

    message = _build_image_task_request(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        target_shape=target_shape,
        image_path=image_path,
        image_mask_path=image_mask_path,
    )

    result_png = await _run_sync_image_task(request, message)
    return _build_openai_response(request, message.task_id, result_png, response_format)
