import asyncio
import time

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from ...schema import (
    SenseNovaVisionTaskRequest,
    SenseNovaVisionTaskResult,
    SenseNovaVisionTaskSubmission,
)
from ...services.generation.sensenova_vision import validate_sensenova_request
from ...task_manager import TaskStatus, task_manager
from ..deps import get_services

router = APIRouter()


def _validate_server_and_message(message: SenseNovaVisionTaskRequest) -> None:
    services = get_services()
    if services.sensenova_vision_service is None:
        raise HTTPException(status_code=503, detail="SenseNova-Vision service is not initialized")
    try:
        services.sensenova_vision_service.ensure_compatible_runner()
        validate_sensenova_request(message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _completed_result(task_id: str) -> dict:
    task_info = task_manager.get_task(task_id)
    if task_info is None or not isinstance(task_info.message, SenseNovaVisionTaskRequest):
        raise HTTPException(status_code=404, detail=f"SenseNova-Vision task not found: {task_id}")
    if task_info.status == TaskStatus.FAILED:
        raise HTTPException(status_code=500, detail=task_info.error or "SenseNova-Vision task failed")
    if task_info.status == TaskStatus.CANCELLED:
        raise HTTPException(status_code=409, detail=task_info.error or "SenseNova-Vision task was cancelled")
    if task_info.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=409, detail=f"SenseNova-Vision task is not completed: {task_info.status.value}")
    result_data = task_manager.get_task_result_data(task_id)
    if result_data is None:
        raise HTTPException(status_code=500, detail=f"SenseNova-Vision result manifest is missing: {task_id}")
    return result_data


async def _wait_for_result(task_id: str, timeout_seconds: int, poll_interval_seconds: float) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        task_info = task_manager.get_task(task_id)
        if task_info is None:
            raise HTTPException(status_code=500, detail=f"SenseNova-Vision task disappeared: {task_id}")
        if task_info.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
            return _completed_result(task_id)
        await asyncio.sleep(poll_interval_seconds)
    task_manager.cancel_task(task_id)
    raise HTTPException(status_code=504, detail=f"SenseNova-Vision task {task_id} timed out after {timeout_seconds} seconds")


async def _watch_disconnect(request: Request, task_id: str, poll_interval_seconds: float = 0.2) -> None:
    while True:
        if await request.is_disconnected():
            task_manager.cancel_task(task_id)
            logger.info(f"SenseNova-Vision client disconnected; cancelled task {task_id}")
            return
        await asyncio.sleep(poll_interval_seconds)


@router.post("/", response_model=SenseNovaVisionTaskSubmission)
async def create_sensenova_vision_task(message: SenseNovaVisionTaskRequest):
    _validate_server_and_message(message)
    try:
        task_id = task_manager.create_task(message)
        message.task_id = task_id
        return SenseNovaVisionTaskSubmission(
            task_id=task_id,
            task_status="pending",
            task=message.task,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/sync", response_model=SenseNovaVisionTaskResult)
async def create_sensenova_vision_task_sync(
    request: Request,
    message: SenseNovaVisionTaskRequest,
    timeout_seconds: int = 1200,
    poll_interval_seconds: float = 0.5,
):
    if timeout_seconds <= 0:
        raise HTTPException(status_code=400, detail="timeout_seconds must be > 0")
    if poll_interval_seconds <= 0:
        raise HTTPException(status_code=400, detail="poll_interval_seconds must be > 0")
    _validate_server_and_message(message)

    try:
        task_id = task_manager.create_task(message)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    message.task_id = task_id
    wait_task = asyncio.create_task(_wait_for_result(task_id, timeout_seconds, poll_interval_seconds))
    disconnect_task = asyncio.create_task(_watch_disconnect(request, task_id))
    try:
        done, pending = await asyncio.wait({wait_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED)
        for pending_task in pending:
            pending_task.cancel()
        if disconnect_task in done:
            if not wait_task.done():
                wait_task.cancel()
            raise HTTPException(status_code=499, detail=f"Client disconnected; SenseNova-Vision task {task_id} cancelled")
        return SenseNovaVisionTaskResult(**wait_task.result())
    except asyncio.CancelledError:
        task_manager.cancel_task(task_id)
        raise


@router.get("/{task_id}/result", response_model=SenseNovaVisionTaskResult)
async def get_sensenova_vision_result(task_id: str):
    return SenseNovaVisionTaskResult(**_completed_result(task_id))
