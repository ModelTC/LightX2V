import asyncio
import base64
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from ..schema import ImageTaskRequest, Usage, hide_request_defaults
from ..services.file_service import FileService
from ..task_manager import TaskStatus, task_manager
from .deps import get_services
from .request_parsing import form_request_openapi, parse_form_request

router = APIRouter()

_SIZE_PATTERN = re.compile(r"^\s*(\d+)\s*x\s*(\d+)\s*$", re.IGNORECASE)
OPENAI_IMAGE_RESULT_POLL_INTERVAL_SECONDS = 0.2
OPENAI_IMAGE_PROTOCOL_FIELDS = {"model", "n", "size", "response_format", "user"}


class OpenAIImageRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra=lambda schema: hide_request_defaults(schema, keep_defaults=OPENAI_IMAGE_PROTOCOL_FIELDS))

    prompt: str = Field(..., description="Text prompt")
    model: Optional[str] = Field(default=None, description="Ignored for compatibility")
    n: int = Field(default=1, description="Number of images, currently only supports 1")
    size: Optional[str] = Field(default=None, description="Image size, e.g. 1024x1024")
    resolution: Optional[int] = Field(None, gt=0, description="Qwen-Image-2.1 reference/automatic output resolution; defaults to the startup config")
    user: Optional[str] = Field(default=None, description="Ignored for compatibility")
    seed: Optional[int] = Field(default=None, description="Optional random seed")


class OpenAIImageGenerationRequest(OpenAIImageRequest):
    response_format: Literal["b64_json"] = "b64_json"


class OpenAIImageEditRequest(OpenAIImageRequest):
    response_format: Literal["url", "b64_json"] = "url"
    negative_prompt: str = ""
    i2i_denoise_strength: Optional[float] = None


class OpenAIImageResponse(BaseModel):
    created: int
    data: Optional[list[dict[str, str]]] = None
    output_format: Optional[Literal["png", "webp", "jpeg"]] = None
    size: Optional[str] = None
    usage: Optional[Usage] = None


def _shape_from_size(size: str) -> tuple[int, int]:
    match = _SIZE_PATTERN.match(size)
    if not match:
        raise ValueError("size must be in WxH format, e.g. 1024x1024")
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError("size width and height must be positive")
    return width, height


async def _wait_task_result_png(task_id: str, timeout_seconds: int, poll_interval_seconds: float) -> tuple[bytes, Usage | dict | None]:
    start_time = time.monotonic()
    status_checks = 0
    while True:
        status_checks += 1
        task_status = task_manager.get_task_status(task_id)
        if not task_status:
            raise HTTPException(status_code=500, detail=f"Task status not found: {task_id}")

        status = task_status.get("status")
        if status == TaskStatus.COMPLETED.value:
            result_png = task_manager.get_task_result_png(task_id)
            usage = task_manager.get_task_result_usage(task_id)
            if result_png:
                wait_elapsed_ms = (time.monotonic() - start_time) * 1000
                completion_observe_lag_ms = 0.0
                end_time = task_status.get("end_time")
                if end_time:
                    completion_observe_lag_ms = (datetime.now() - end_time).total_seconds() * 1000
                logger.info(
                    f"Task {task_id} OpenAI image wait_task_result cost total={wait_elapsed_ms:.2f} ms "
                    f"completion_observe_lag={completion_observe_lag_ms:.2f} ms "
                    f"poll_interval={poll_interval_seconds:.2f} s status_checks={status_checks}"
                )
                return result_png, usage
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


async def _run_sync_image_task(request: Request, message: ImageTaskRequest) -> tuple[bytes, Usage | dict | None]:
    task_id = None
    timeout_seconds = 600
    poll_interval_seconds = OPENAI_IMAGE_RESULT_POLL_INTERVAL_SECONDS

    try:
        message._prefer_memory_result = True
        create_task_start = time.perf_counter()
        task_id = task_manager.create_task(message)
        create_task_elapsed_ms = (time.perf_counter() - create_task_start) * 1000
        message.task_id = task_id
        logger.info(f"Task {task_id} OpenAI image create_task cost {create_task_elapsed_ms:.2f} ms prompt_chars={len(message.prompt)} image_size={message.size}")

        wait_task = asyncio.create_task(_wait_task_result_png(task_id, timeout_seconds, poll_interval_seconds))
        disconnect_task = asyncio.create_task(_watch_client_disconnect(request, task_id))

        done, pending = await asyncio.wait({wait_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED)
        for pending_task in pending:
            pending_task.cancel()

        if disconnect_task in done:
            if not wait_task.done():
                wait_task.cancel()
            raise HTTPException(status_code=499, detail=f"Client disconnected, task {task_id} cancelled")

        result_png, usage = wait_task.result()
        logger.info(f"Task {task_id} OpenAI image task result ready, building response")
        return result_png, usage
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
    output_path.write_bytes(image_bytes)

    base = str(request.base_url).rstrip("/")
    return f"{base}/v1/files/download/{file_name}"


def _build_openai_response(request: Request, task_id: str, image_bytes: bytes, response_format: Literal["url", "b64_json"], usage: Usage | dict | None = None, size: Optional[str] = None):
    total_start = time.perf_counter()
    usage_obj = Usage(**usage) if isinstance(usage, dict) else usage
    if response_format == "b64_json":
        base64_start = time.perf_counter()
        b64_json = base64.b64encode(image_bytes).decode("utf-8")
        base64_elapsed_ms = (time.perf_counter() - base64_start) * 1000
        response = OpenAIImageResponse(created=int(time.time()), data=[{"b64_json": b64_json}], output_format="png", usage=usage_obj, size=size)
        total_elapsed_ms = (time.perf_counter() - total_start) * 1000
        logger.info(
            f"Task {task_id} OpenAI image response build cost total={total_elapsed_ms:.2f} ms base64={base64_elapsed_ms:.2f} ms format=b64_json png_bytes={len(image_bytes)} b64_chars={len(b64_json)}"
        )
        return response

    url_start = time.perf_counter()
    url = _build_url_response(request, task_id, image_bytes)
    url_elapsed_ms = (time.perf_counter() - url_start) * 1000
    response = OpenAIImageResponse(created=int(time.time()), data=[{"url": url}], output_format="png", usage=usage_obj, size=size)
    total_elapsed_ms = (time.perf_counter() - total_start) * 1000
    logger.info(f"Task {task_id} OpenAI image response build cost total={total_elapsed_ms:.2f} ms url_write={url_elapsed_ms:.2f} ms format=url png_bytes={len(image_bytes)}")
    return response


def _build_image_task_request(body: OpenAIImageRequest, *, task: str) -> ImageTaskRequest:
    if body.n != 1:
        raise HTTPException(status_code=400, detail="Only n=1 is currently supported")
    if not body.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    # The OpenAI schema is the public field allowlist. Only protocol fields need
    # special handling; explicitly submitted inference fields pass through.
    payload = body.model_dump(exclude_unset=True, exclude_none=True, exclude=OPENAI_IMAGE_PROTOCOL_FIELDS)
    if body.size:
        try:
            width, height = _shape_from_size(body.size)
            payload["size"] = [height, width]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return ImageTaskRequest(task=task, **payload)


@router.post("/generations", response_model=OpenAIImageResponse)
async def create_openai_image_generation(request: Request, body: OpenAIImageGenerationRequest):
    message = _build_image_task_request(body, task="t2i")
    result_png, usage = await _run_sync_image_task(request, message)
    return _build_openai_response(request, message.task_id, result_png, body.response_format, usage, size=body.size)


async def _save_upload_file(file: UploadFile, file_service: FileService) -> str:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename")

    filename = file.filename if Path(file.filename).suffix else file.filename + ".png"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"Uploaded file is empty: {file.filename}")
    file_path = await asyncio.to_thread(file_service.save_uploaded_file, content, filename)
    return str(file_path)


@router.post(
    "/edits",
    response_model=OpenAIImageResponse,
    openapi_extra=form_request_openapi(
        OpenAIImageEditRequest,
        image={"type": "array", "items": {"type": "string", "format": "binary"}},
        mask={"type": "string", "format": "binary"},
    ),
)
async def create_openai_image_edit(
    request: Request,
    image: list[UploadFile] | None = File(default=None),
    mask: UploadFile | None = File(default=None),
):
    form = await request.form()
    image_uploads = list(image or [])
    if not image_uploads:
        image_uploads = [upload for upload in form.getlist("image[]") if hasattr(upload, "filename") and hasattr(upload, "read")]

    # Preserve FastAPI's previous empty-form defaults, except negative_prompt:
    # its explicitly submitted empty string must still reach the runner.
    form_data = {key: value for key, value in form.items() if key not in {"image", "image[]", "mask"} and (value != "" or key == "negative_prompt")}
    body = parse_form_request(OpenAIImageEditRequest, form_data)
    message = _build_image_task_request(body, task="i2i")

    services = get_services()
    assert services.file_service is not None, "File service is not initialized"

    if not image_uploads:
        raise HTTPException(status_code=400, detail="At least one image is required")
    if mask is not None and len(image_uploads) != 1:
        raise HTTPException(status_code=400, detail="mask is only supported when exactly one image is uploaded")

    image_paths = []
    for image_upload in image_uploads:
        image_paths.append(await _save_upload_file(image_upload, services.file_service))
    message.image_path = ",".join(image_paths)

    if mask is not None:
        message.image_mask_path = await _save_upload_file(mask, services.file_service)

    result_png, usage = await _run_sync_image_task(request, message)
    return _build_openai_response(request, message.task_id, result_png, body.response_format, usage, size=body.size)
