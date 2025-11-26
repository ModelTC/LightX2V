import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from ...schema import ImageTaskRequest, TaskResponse
from ...task_manager import task_manager
from ..deps import get_services, validate_url_async

router = APIRouter()


def _write_file_sync(file_path: Path, content: bytes) -> None:
    with open(file_path, "wb") as buffer:
        buffer.write(content)


@router.post("/", response_model=TaskResponse)
async def create_image_task(message: ImageTaskRequest):
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
        logger.error(f"Failed to create image task: {e}")
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
        logger.error(f"Failed to create image form task: {e}")
        raise HTTPException(status_code=500, detail=str(e))
