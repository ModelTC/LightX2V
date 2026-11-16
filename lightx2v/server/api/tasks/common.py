import gc
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from loguru import logger

from ...schema import StopTaskResponse, TaskResult, TaskResults
from ...task_manager import TaskStatus, task_manager
from ..deps import get_services
from ..files import _stream_file_response

router = APIRouter()


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


def _completed_task(task_id):
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=409, detail=task.error or "Task not completed")
    return task


@router.get("/{task_id}/results", response_model=TaskResults)
async def get_task_results(task_id: str):
    task = _completed_task(task_id)
    return TaskResults(
        task_id=task_id,
        results=[TaskResult(name=output.name, type=output.type, media_type=output.media_type, url=f"/v1/tasks/{task_id}/results/{output.name}") for output in task.results],
    )


@router.get("/{task_id}/results/{name}")
async def download_task_result(task_id: str, name: str):
    task = _completed_task(task_id)
    output = next((output for output in task.results if output.name == name), None)
    if output is None:
        raise HTTPException(status_code=404, detail="Result not found")
    content = output.content
    if content is not None:
        return Response(content=content, media_type="image/png" if output.type == "image" else output.media_type)
    if output.path is None:
        raise HTTPException(status_code=410, detail="Result is no longer in memory; submit a new task to regenerate it")
    path = Path(output.path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found")
    return FileResponse(path, media_type=output.media_type, filename=path.name)


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
