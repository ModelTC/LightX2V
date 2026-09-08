import asyncio

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from ...schema import TaskResponse, VideoTaskRequest
from ...task_manager import task_manager
from ..deps import get_services, validate_url_async

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


@router.post("/form", response_model=TaskResponse)
async def create_video_task_form(
    image_file: UploadFile = File(...),
    last_frame_file: UploadFile = File(None),
    prompt: str = Form(default=""),
    save_result_path: str = Form(default=""),
    negative_prompt: str = Form(default=""),
    infer_steps: int = Form(default=5),
    target_video_length: int = Form(default=81),
    seed: int = Form(default=42),
    audio_file: UploadFile = File(None),
    video_duration: int = Form(default=5),
    target_fps: int = Form(default=16),
):
    services = get_services()
    assert services.file_service is not None, "File service is not initialized"

    image_path = ""
    if image_file and image_file.filename:
        content = await image_file.read()
        image_path = str(await asyncio.to_thread(services.file_service.save_uploaded_file, content, image_file.filename))

    last_frame_path = ""
    if last_frame_file and last_frame_file.filename:
        content = await last_frame_file.read()
        last_frame_path = str(await asyncio.to_thread(services.file_service.save_uploaded_file, content, last_frame_file.filename))

    audio_path = ""
    if audio_file and audio_file.filename:
        content = await audio_file.read()
        audio_path = str(await asyncio.to_thread(services.file_service.save_uploaded_file, content, audio_file.filename, services.file_service.input_audio_dir))

    message = VideoTaskRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        last_frame_path=last_frame_path,
        image_path=image_path,
        save_result_path=save_result_path,
        infer_steps=infer_steps,
        target_video_length=target_video_length,
        seed=seed,
        audio_path=audio_path,
        video_duration=video_duration,
        target_fps=target_fps,
    )

    return await create_video_task(message)
