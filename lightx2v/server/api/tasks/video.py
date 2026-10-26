import asyncio

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from loguru import logger

from ...schema import TaskResponse, VideoTaskRequest
from ...task_manager import task_manager
from ..deps import get_services, validate_url_async
from ..request_parsing import form_request_openapi, parse_form_request

router = APIRouter()


@router.post("/", response_model=TaskResponse)
async def create_video_task(message: VideoTaskRequest):
    try:
        if hasattr(message, "image_path") and message.image_path and message.image_path.startswith("http"):
            if not await validate_url_async(message.image_path):
                raise HTTPException(status_code=400, detail=f"Image URL is not accessible: {message.image_path}")

        task_id = task_manager.create_task(message)
        message.task_id = task_id

        return TaskResponse(
            task_id=task_id,
            task_status="pending",
            save_result_path=message.save_result_path,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create video task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/form",
    response_model=TaskResponse,
    openapi_extra=form_request_openapi(
        VideoTaskRequest,
        image_file={"type": "string", "format": "binary"},
        last_frame_file={"type": "string", "format": "binary"},
        audio_file={"type": "string", "format": "binary"},
    ),
)
async def create_video_task_form(
    request: Request,
    image_file: UploadFile = File(None),
    last_frame_file: UploadFile = File(None),
    audio_file: UploadFile = File(None),
):
    services = get_services()
    assert services.file_service is not None, "File service is not initialized"

    form = await request.form()
    message = parse_form_request(VideoTaskRequest, {key: value for key, value in form.items() if key not in {"image_file", "last_frame_file", "audio_file"}})
    if image_file and image_file.filename:
        content = await image_file.read()
        message.image_path = str(await asyncio.to_thread(services.file_service.save_uploaded_file, content, image_file.filename))

    if last_frame_file and last_frame_file.filename:
        content = await last_frame_file.read()
        message.last_frame_path = str(await asyncio.to_thread(services.file_service.save_uploaded_file, content, last_frame_file.filename))

    if audio_file and audio_file.filename:
        content = await audio_file.read()
        message.audio_path = str(await asyncio.to_thread(services.file_service.save_uploaded_file, content, audio_file.filename, services.file_service.input_audio_dir))

    return await create_video_task(message)
