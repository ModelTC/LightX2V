import asyncio
import gc
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
import torch
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger

from .schema import (
    StopTaskResponse,
    TaskRequest,
    TaskResponse,
)
from .service import DistributedInferenceService, FileService, VideoGenerationService
from .task_manager import TaskStatus, task_manager


class ApiServer:
    def __init__(self, max_queue_size: int = 10, app: Optional[FastAPI] = None):
        self.app = app or FastAPI(title="LightX2V API", version="1.0.0")
        self.file_service = None
        self.inference_service = None
        self.video_service = None
        self.max_queue_size = max_queue_size

        self.processing_thread = None
        self.stop_processing = threading.Event()

        self.tasks_router = APIRouter(prefix="/v1/tasks", tags=["tasks"])
        self.files_router = APIRouter(prefix="/v1/files", tags=["files"])
        self.service_router = APIRouter(prefix="/v1/service", tags=["service"])

        self._setup_routes()

    def _setup_routes(self):
        @self.app.get("/")
        def redirect_to_docs():
            return RedirectResponse(url="/docs")

        self._setup_task_routes()
        self._setup_file_routes()
        self._setup_service_routes()

        self.app.include_router(self.tasks_router)
        self.app.include_router(self.files_router)
        self.app.include_router(self.service_router)

    def _write_file_sync(self, file_path: Path, content: bytes) -> None:
        with open(file_path, "wb") as buffer:
            buffer.write(content)

    def _stream_file_response(self, file_path: Path, filename: str | None = None) -> StreamingResponse:
        assert self.file_service is not None, "File service is not initialized"

        try:
            resolved_path = file_path.resolve()

            if not str(resolved_path).startswith(str(self.file_service.output_video_dir.resolve())):
                raise HTTPException(status_code=403, detail="Access to this file is not allowed")

            if not resolved_path.exists() or not resolved_path.is_file():
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

            file_size = resolved_path.stat().st_size
            actual_filename = filename or resolved_path.name

            # Set appropriate MIME type
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

    def _setup_task_routes(self):
        @self.tasks_router.post("/", response_model=TaskResponse)
        async def create_task(message: TaskRequest):
            """Create video generation task"""
            try:
                if hasattr(message, "image_path") and message.image_path and message.image_path.startswith("http"):
                    if not await self._validate_image_url(message.image_path):
                        raise HTTPException(status_code=400, detail=f"Image URL is not accessible: {message.image_path}")

                task_id = task_manager.create_task(message)
                message.task_id = task_id

                self._ensure_processing_thread_running()

                return TaskResponse(
                    task_id=task_id,
                    task_status="pending",
                    save_result_path=message.save_result_path,
                )
            except RuntimeError as e:
                raise HTTPException(status_code=503, detail=str(e))
            except Exception as e:
                logger.error(f"Failed to create task: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.tasks_router.post("/form", response_model=TaskResponse)
        async def create_task_form(
            image_file: UploadFile = File(...),
            prompt: str = Form(default=""),
            save_result_path: str = Form(default=""),
            use_prompt_enhancer: bool = Form(default=False),
            negative_prompt: str = Form(default=""),
            num_fragments: int = Form(default=1),
            infer_steps: int = Form(default=5),
            target_video_length: int = Form(default=81),
            seed: int = Form(default=42),
            audio_file: UploadFile = File(None),
            video_duration: int = Form(default=5),
        ):
            assert self.file_service is not None, "File service is not initialized"

            async def save_file_async(file: UploadFile, target_dir: Path) -> str:
                if not file or not file.filename:
                    return ""

                file_extension = Path(file.filename).suffix
                unique_filename = f"{uuid.uuid4()}{file_extension}"
                file_path = target_dir / unique_filename

                content = await file.read()

                await asyncio.to_thread(self._write_file_sync, file_path, content)

                return str(file_path)

            image_path = ""
            if image_file and image_file.filename:
                image_path = await save_file_async(image_file, self.file_service.input_image_dir)

            audio_path = ""
            if audio_file and audio_file.filename:
                audio_path = await save_file_async(audio_file, self.file_service.input_audio_dir)

            message = TaskRequest(
                prompt=prompt,
                use_prompt_enhancer=use_prompt_enhancer,
                negative_prompt=negative_prompt,
                image_path=image_path,
                num_fragments=num_fragments,
                save_result_path=save_result_path,
                infer_steps=infer_steps,
                target_video_length=target_video_length,
                seed=seed,
                audio_path=audio_path,
                video_duration=video_duration,
            )

            try:
                task_id = task_manager.create_task(message)
                message.task_id = task_id

                self._ensure_processing_thread_running()

                return TaskResponse(
                    task_id=task_id,
                    task_status="pending",
                    save_result_path=message.save_result_path,
                )
            except RuntimeError as e:
                raise HTTPException(status_code=503, detail=str(e))
            except Exception as e:
                logger.error(f"Failed to create form task: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.tasks_router.get("/", response_model=dict)
        async def list_tasks():
            return task_manager.get_all_tasks()

        @self.tasks_router.get("/queue/status", response_model=dict)
        async def get_queue_status():
            service_status = task_manager.get_service_status()
            return {
                "is_processing": task_manager.is_processing(),
                "current_task": service_status.get("current_task"),
                "pending_count": task_manager.get_pending_task_count(),
                "active_count": task_manager.get_active_task_count(),
                "queue_size": self.max_queue_size,
                "queue_available": self.max_queue_size - task_manager.get_active_task_count(),
            }

        @self.tasks_router.get("/{task_id}/status")
        async def get_task_status(task_id: str):
            status = task_manager.get_task_status(task_id)
            if not status:
                raise HTTPException(status_code=404, detail="Task not found")
            return status

        @self.tasks_router.get("/{task_id}/result")
        async def get_task_result(task_id: str):
            assert self.video_service is not None, "Video service is not initialized"
            assert self.file_service is not None, "File service is not initialized"

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
                    full_path = self.file_service.output_video_dir / save_result_path

                return self._stream_file_response(full_path)

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error occurred while getting task result: {e}")
                raise HTTPException(status_code=500, detail="Failed to get task result")

        @self.tasks_router.delete("/{task_id}", response_model=StopTaskResponse)
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

        @self.tasks_router.delete("/all/running", response_model=StopTaskResponse)
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

    def _setup_file_routes(self):
        @self.files_router.get("/download/{file_path:path}")
        async def download_file(file_path: str):
            assert self.file_service is not None, "File service is not initialized"

            try:
                full_path = self.file_service.output_video_dir / file_path
                return self._stream_file_response(full_path)
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error occurred while processing file download request: {e}")
                raise HTTPException(status_code=500, detail="File download failed")

    def _setup_service_routes(self):
        @self.service_router.get("/status", response_model=dict)
        async def get_service_status():
            return task_manager.get_service_status()

        @self.service_router.get("/metadata", response_model=dict)
        async def get_service_metadata():
            assert self.inference_service is not None, "Inference service is not initialized"
            return self.inference_service.server_metadata()

    async def _validate_image_url(self, image_url: str) -> bool:
        if not image_url or not image_url.startswith("http"):
            return True

        try:
            parsed_url = urlparse(image_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return False

            timeout = httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)
            async with httpx.AsyncClient(verify=False, timeout=timeout) as client:
                response = await client.head(image_url, follow_redirects=True)
                return response.status_code < 400
        except Exception as e:
            logger.warning(f"URL validation failed for {image_url}: {str(e)}")
            return False

    def _ensure_processing_thread_running(self):
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_processing.clear()
            self.processing_thread = threading.Thread(target=self._task_processing_loop, daemon=True)
            self.processing_thread.start()
            logger.info("Started task processing thread")

    def _task_processing_loop(self):
        logger.info("Task processing loop started")

        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()

        while not self.stop_processing.is_set():
            task_id = task_manager.get_next_pending_task()

            if task_id is None:
                time.sleep(1)
                continue

            task_info = task_manager.get_task(task_id)
            if task_info and task_info.status == TaskStatus.PENDING:
                logger.info(f"Processing task {task_id}")
                loop.run_until_complete(self._process_single_task(task_info))

        loop.close()
        logger.info("Task processing loop stopped")

    async def _process_single_task(self, task_info: Any):
        assert self.video_service is not None, "Video service is not initialized"

        task_id = task_info.task_id
        message = task_info.message

        lock_acquired = task_manager.acquire_processing_lock(task_id, timeout=1)
        if not lock_acquired:
            logger.error(f"Task {task_id} failed to acquire processing lock")
            task_manager.fail_task(task_id, "Failed to acquire processing lock")
            return

        try:
            task_manager.start_task(task_id)

            if task_info.stop_event.is_set():
                logger.info(f"Task {task_id} cancelled before processing")
                task_manager.fail_task(task_id, "Task cancelled")
                return

            result = await self.video_service.generate_video_with_stop_event(message, task_info.stop_event)

            if result:
                task_manager.complete_task(task_id, result.save_result_path)
                logger.info(f"Task {task_id} completed successfully")
            else:
                if task_info.stop_event.is_set():
                    task_manager.fail_task(task_id, "Task cancelled during processing")
                    logger.info(f"Task {task_id} cancelled during processing")
                else:
                    task_manager.fail_task(task_id, "Generation failed")
                    logger.error(f"Task {task_id} generation failed")

        except Exception as e:
            logger.exception(f"Task {task_id} processing failed: {str(e)}")
            task_manager.fail_task(task_id, str(e))
        finally:
            if lock_acquired:
                task_manager.release_processing_lock(task_id)

    def initialize_services(self, cache_dir: Path, inference_service: DistributedInferenceService):
        self.file_service = FileService(cache_dir)
        self.inference_service = inference_service
        self.video_service = VideoGenerationService(self.file_service, inference_service)

    async def cleanup(self):
        self.stop_processing.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)

        if self.file_service:
            await self.file_service.cleanup()

    def get_app(self) -> FastAPI:
        return self.app
