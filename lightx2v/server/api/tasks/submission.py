"""Shared task submission and synchronous image waiting."""

import asyncio
import time

from fastapi import HTTPException

from ...schema import TaskResponse
from ...task_manager import TaskStatus, task_manager
from ..deps import validate_image_urls


async def submit_task(message, *, required_output=None, memory_result=False):
    await validate_image_urls(message)
    message._required_output = required_output
    message._prefer_memory_result = memory_result
    try:
        task_id = task_manager.create_task(message)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return TaskResponse(task_id=task_id, task_status="pending", save_result_path=message.get("save_result_path"))


async def _wait_image(task_id, timeout_seconds, poll_interval_seconds):
    started = time.monotonic()
    while True:
        status = task_manager.get_task_status(task_id)
        if status is None:
            raise HTTPException(status_code=500, detail=f"Task status not found: {task_id}")
        if status["status"] == TaskStatus.COMPLETED.value:
            png = task_manager.get_task_result_png(task_id)
            if png is None:
                raise HTTPException(status_code=500, detail="Task completed without an in-memory image")
            return png, task_manager.get_task_result_usage(task_id)
        if status["status"] == TaskStatus.FAILED.value:
            code = 413 if status.get("error_type") == "ValueError" else 500
            raise HTTPException(status_code=code, detail=status.get("error", "Task failed"))
        if status["status"] == TaskStatus.CANCELLED.value:
            raise HTTPException(status_code=409, detail="Task cancelled")
        if time.monotonic() - started > timeout_seconds:
            task_manager.cancel_task(task_id)
            raise HTTPException(status_code=504, detail=f"Task {task_id} timed out after {timeout_seconds} seconds")
        await asyncio.sleep(poll_interval_seconds)


async def _watch_disconnect(request, task_id):
    while not await request.is_disconnected():
        await asyncio.sleep(0.2)
    task_manager.cancel_task(task_id)


async def run_image_task(request, message, *, timeout_seconds=600, poll_interval_seconds=0.5):
    if timeout_seconds <= 0 or poll_interval_seconds <= 0:
        raise HTTPException(status_code=400, detail="timeout_seconds and poll_interval_seconds must be > 0")
    await submit_task(message, required_output="image", memory_result=True)
    waiting = asyncio.create_task(_wait_image(message.task_id, timeout_seconds, poll_interval_seconds))
    disconnected = asyncio.create_task(_watch_disconnect(request, message.task_id))
    try:
        done, _ = await asyncio.wait({waiting, disconnected}, return_when=asyncio.FIRST_COMPLETED)
        if disconnected in done:
            raise HTTPException(status_code=499, detail=f"Client disconnected, task {message.task_id} cancelled")
        return waiting.result()
    except asyncio.CancelledError:
        task_manager.cancel_task(message.task_id)
        raise
    finally:
        waiting.cancel()
        disconnected.cancel()
