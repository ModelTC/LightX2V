import gc
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from ...schema import StopTaskResponse
from ...task_manager import TaskStatus, task_manager
from ..deps import get_services

router = APIRouter()


def _stream_file_response(file_path: Path, filename: str | None = None) -> StreamingResponse:
    services = get_services()
    assert services.file_service is not None, "File service is not initialized"

    try:
        resolved_path = file_path.resolve()

        if not str(resolved_path).startswith(str(services.file_service.output_video_dir.resolve())):
            raise HTTPException(status_code=403, detail="Access to this file is not allowed")

        if not resolved_path.exists() or not resolved_path.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        file_size = resolved_path.stat().st_size
        actual_filename = filename or resolved_path.name

        mime_type = "application/octet-stream"
        if actual_filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            mime_type = "video/mp4"
        elif actual_filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            mime_type = "image/jpeg"

        headers = {
            "Content-Disposition": f'attachment; filename="{actual_filename}"',
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
        }

        def file_stream_generator(file_path: str, chunk_size: int = 1024 * 1024):
            with open(file_path, "rb") as file:
                while chunk := file.read(chunk_size):
                    yield chunk

        return StreamingResponse(
            file_stream_generator(str(resolved_path)),
            media_type=mime_type,
            headers=headers,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error occurred while processing file stream response: {e}")
        raise HTTPException(status_code=500, detail="File transfer failed")


@router.get("/")
async def list_tasks():
    return task_manager.get_all_tasks()


@router.get("/queue/status")
async def get_queue_status():
    services = get_services()
    service_status = task_manager.get_service_status()
    return {
        "is_processing": task_manager.is_processing(),
        "current_task": service_status.get("current_task"),
        "pending_count": task_manager.get_pending_task_count(),
        "active_count": task_manager.get_active_task_count(),
        "queue_size": services.max_queue_size,
        "queue_available": services.max_queue_size - task_manager.get_active_task_count(),
    }


@router.get("/{task_id}/status")
async def get_task_status(task_id: str):
    status = task_manager.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


@router.get("/{task_id}/result")
async def get_task_result(task_id: str):
    services = get_services()
    assert services.video_service is not None, "Video service is not initialized"
    assert services.file_service is not None, "File service is not initialized"

    try:
        task_status = task_manager.get_task_status(task_id)

        if not task_status:
            raise HTTPException(status_code=404, detail="Task not found")

        if task_status.get("status") != TaskStatus.COMPLETED.value:
            raise HTTPException(status_code=404, detail="Task not completed")

        save_result_path = task_status.get("save_result_path")
        if not save_result_path:
            raise HTTPException(status_code=404, detail="Task result file does not exist")

        full_path = Path(save_result_path)
        if not full_path.is_absolute():
            full_path = services.file_service.output_video_dir / save_result_path

        return _stream_file_response(full_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error occurred while getting task result: {e}")
        raise HTTPException(status_code=500, detail="Failed to get task result")


@router.delete("/{task_id}", response_model=StopTaskResponse)
async def stop_task(task_id: str):
    try:
        if task_manager.cancel_task(task_id):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Task {task_id} stopped successfully.")
            return StopTaskResponse(stop_status="success", reason="Task stopped successfully.")
        else:
            return StopTaskResponse(stop_status="do_nothing", reason="Task not found or already completed.")
    except Exception as e:
        logger.error(f"Error occurred while stopping task {task_id}: {str(e)}")
        return StopTaskResponse(stop_status="error", reason=str(e))


@router.delete("/all/running", response_model=StopTaskResponse)
async def stop_all_running_tasks():
    try:
        task_manager.cancel_all_tasks()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All tasks stopped successfully.")
        return StopTaskResponse(stop_status="success", reason="All tasks stopped successfully.")
    except Exception as e:
        logger.error(f"Error occurred while stopping all tasks: {str(e)}")
        return StopTaskResponse(stop_status="error", reason=str(e))
