import asyncio
import base64
import re
import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from lightx2v.models.runners.registry import get_model_spec
from lightx2v.utils.request_spec import create_request_model, is_list_type

from ..schema import TaskRequest, Usage, hide_request_defaults
from ..services.file_service import FileService
from .deps import get_services
from .request_parsing import form_request_openapi, parse_form_request, validate_request
from .tasks.submission import run_image_task

_SIZE_PATTERN = re.compile(r"^\s*(\d+)\s*x\s*(\d+)\s*$", re.IGNORECASE)
OPENAI_IMAGE_RESULT_POLL_INTERVAL_SECONDS = 0.2
OPENAI_IMAGE_PROTOCOL_FIELDS = {"model", "n", "size", "response_format", "user"}


class OpenAIImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra=lambda schema: hide_request_defaults(schema, keep_defaults=OPENAI_IMAGE_PROTOCOL_FIELDS))

    prompt: str = Field(..., description="Text prompt")
    model: Optional[str] = Field(default=None, description="Ignored for compatibility")
    n: int = Field(default=1, description="Number of images, currently only supports 1")
    size: Optional[str] = Field(default=None, description="Image size, e.g. 1024x1024")
    user: Optional[str] = Field(default=None, description="Ignored for compatibility")


class OpenAIImageGenerationRequest(OpenAIImageRequest):
    response_format: Literal["b64_json"] = "b64_json"


class OpenAIImageEditRequest(OpenAIImageRequest):
    response_format: Literal["url", "b64_json"] = "url"


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


def _build_image_task_request(body: OpenAIImageRequest, request_type, *, task: str | None = None) -> TaskRequest:
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
    if task is not None:
        payload["task"] = task
    return validate_request(request_type, payload)


async def _save_upload_file(file: UploadFile, file_service: FileService) -> str:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename")

    filename = file.filename if Path(file.filename).suffix else file.filename + ".png"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"Uploaded file is empty: {file.filename}")
    file_path = await asyncio.to_thread(file_service.save_uploaded_file, content, filename)
    return str(file_path)


def create_router(config, request_type):
    router = APIRouter()
    spec = get_model_spec(config["model_cls"])
    tasks = spec.get_tasks(config)
    multi_task = len(spec.get_tasks(config)) > 1

    def request_model(task, base):
        arguments = {name: argument for name, argument in spec.get_arguments(config, task).items() if argument.server and name != "task" and name not in base.model_fields}
        return create_request_model(f"{config['model_cls']}_{base.__name__}", arguments, base=base)

    if "t2i" in tasks:
        generation_request = request_model("t2i", OpenAIImageGenerationRequest)

        @router.post("/generations", response_model=OpenAIImageResponse)
        async def create_openai_image_generation(request: Request, body: generation_request):
            message = _build_image_task_request(body, request_type, task="t2i" if multi_task else None)
            result_png, usage = await run_image_task(request, message, poll_interval_seconds=OPENAI_IMAGE_RESULT_POLL_INTERVAL_SECONDS)
            return _build_openai_response(request, message.task_id, result_png, body.response_format, usage, size=body.size)

    if "i2i" in tasks and "image_path" in spec.get_fields(config, "i2i"):
        edit_request = request_model("i2i", OpenAIImageEditRequest)
        multiple_images = is_list_type(spec.get_arguments(config, "i2i")["image_path"].type)

        @router.post(
            "/edits",
            response_model=OpenAIImageResponse,
            openapi_extra=form_request_openapi(
                edit_request,
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
            body = parse_form_request(edit_request, form_data)
            message = _build_image_task_request(body, request_type, task="i2i" if multi_task else None)

            services = get_services()
            assert services.file_service is not None, "File service is not initialized"

            if not image_uploads:
                raise HTTPException(status_code=400, detail="At least one image is required")
            if not multiple_images and len(image_uploads) > 1:
                raise HTTPException(status_code=400, detail="This model accepts only one input image")
            if mask is not None and len(image_uploads) != 1:
                raise HTTPException(status_code=400, detail="mask is only supported when exactly one image is uploaded")

            image_paths = []
            for image_upload in image_uploads:
                image_paths.append(await _save_upload_file(image_upload, services.file_service))
            message.image_path = image_paths if multiple_images else image_paths[0]

            if mask is not None:
                message.image_mask_path = await _save_upload_file(mask, services.file_service)

            result_png, usage = await run_image_task(request, message, poll_interval_seconds=OPENAI_IMAGE_RESULT_POLL_INTERVAL_SECONDS)
            return _build_openai_response(request, message.task_id, result_png, body.response_format, usage, size=body.size)

    return router
