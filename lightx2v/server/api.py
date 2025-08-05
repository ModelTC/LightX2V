import asyncio
import gc
import threading
import uuid
from pathlib import Path
from typing import Optional

import torch
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger

from .schema import (
    ServiceStatusResponse,
    StopTaskResponse,
    TaskRequest,
    TaskResponse,
)
from .service import DistributedInferenceService, FileService, VideoGenerationService
from .utils import ServiceStatus


class ApiServer:
    def __init__(self):
        self.app = FastAPI(title="LightX2V API", version="1.0.0")
        self.file_service = None
        self.inference_service = None
        self.video_service = None
        self.thread = None
        self.stop_generation_event = threading.Event()

        # Create routers
        self.tasks_router = APIRouter(prefix="/v1/tasks", tags=["tasks"])
        self.files_router = APIRouter(prefix="/v1/files", tags=["files"])
        self.service_router = APIRouter(prefix="/v1/service", tags=["service"])

        self._setup_routes()

    def _setup_routes(self):
        """Setup routes"""
        self._setup_task_routes()
        self._setup_file_routes()
        self._setup_service_routes()

        # Register routers
        self.app.include_router(self.tasks_router)
        self.app.include_router(self.files_router)
        self.app.include_router(self.service_router)

    def _write_file_sync(self, file_path: Path, content: bytes) -> None:
        """同步写入文件到指定路径"""
        with open(file_path, "wb") as buffer:
            buffer.write(content)

    def _stream_file_response(self, file_path: Path, filename: str | None = None) -> StreamingResponse:
        """Common file streaming response method"""
        assert self.file_service is not None, "File service is not initialized"

        try:
            resolved_path = file_path.resolve()

            # Security check: ensure file is within allowed directory
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
                task_id = ServiceStatus.start_task(message)

                # Use background thread to handle long-running tasks
                self.stop_generation_event.clear()
                self.thread = threading.Thread(
                    target=self._process_video_generation,
                    args=(message, self.stop_generation_event),
                    daemon=True,
                )
                self.thread.start()

                return TaskResponse(
                    task_id=task_id,
                    task_status="processing",
                    save_video_path=message.save_video_path,
                )
            except RuntimeError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.tasks_router.post("/form", response_model=TaskResponse)
        async def create_task_form(
            image_file: UploadFile = File(...),
            prompt: str = Form(default=""),
            save_video_path: str = Form(default=""),
            use_prompt_enhancer: bool = Form(default=False),
            negative_prompt: str = Form(default=""),
            num_fragments: int = Form(default=1),
            infer_steps: int = Form(default=5),
            target_video_length: int = Form(default=81),
            seed: int = Form(default=42),
            audio_file: Optional[UploadFile] = File(default=None),
            video_duration: int = Form(default=5),
        ):
            """Create video generation task via form"""
            assert self.file_service is not None, "File service is not initialized"

            async def save_file_async(file: UploadFile, target_dir: Path) -> str:
                """异步保存文件到指定目录"""
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
                save_video_path=save_video_path,
                infer_steps=infer_steps,
                target_video_length=target_video_length,
                seed=seed,
                audio_path=audio_path,
                video_duration=video_duration,
            )

            try:
                task_id = ServiceStatus.start_task(message)
                self.stop_generation_event.clear()
                self.thread = threading.Thread(
                    target=self._process_video_generation,
                    args=(message, self.stop_generation_event),
                    daemon=True,
                )
                self.thread.start()

                return TaskResponse(
                    task_id=task_id,
                    task_status="processing",
                    save_video_path=message.save_video_path,
                )
            except RuntimeError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.tasks_router.get("/", response_model=dict)
        async def list_tasks():
            """Get all task list"""
            return ServiceStatus.get_all_tasks()

        @self.tasks_router.get("/{task_id}/status")
        async def get_task_status(task_id: str):
            """Get status of specified task"""
            return ServiceStatus.get_status_task_id(task_id)

        @self.tasks_router.get("/{task_id}/result")
        async def get_task_result(task_id: str):
            """Get result video file of specified task"""
            assert self.video_service is not None, "Video service is not initialized"
            assert self.file_service is not None, "File service is not initialized"

            try:
                task_status = ServiceStatus.get_status_task_id(task_id)

                if not task_status or task_status.get("status") != "completed":
                    raise HTTPException(status_code=404, detail="Task not completed or does not exist")

                save_video_path = task_status.get("save_video_path")
                if not save_video_path:
                    raise HTTPException(status_code=404, detail="Task result file does not exist")

                full_path = Path(save_video_path)
                if not full_path.is_absolute():
                    full_path = self.file_service.output_video_dir / save_video_path

                return self._stream_file_response(full_path)

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error occurred while getting task result: {e}")
                raise HTTPException(status_code=500, detail="Failed to get task result")

        @self.tasks_router.delete("/running", response_model=StopTaskResponse)
        async def stop_running_task():
            """Stop currently running task"""
            if self.thread and self.thread.is_alive():
                try:
                    logger.info("Sending stop signal to running task thread...")
                    self.stop_generation_event.set()
                    self.thread.join(timeout=5)

                    if self.thread.is_alive():
                        logger.warning("Task thread did not stop within the specified time, manual intervention may be required.")
                        return StopTaskResponse(
                            stop_status="warning",
                            reason="Task thread did not stop within the specified time, manual intervention may be required.",
                        )
                    else:
                        self.thread = None
                        ServiceStatus.clean_stopped_task()
                        gc.collect()
                        torch.cuda.empty_cache()
                        logger.info("Task stopped successfully.")
                        return StopTaskResponse(stop_status="success", reason="Task stopped successfully.")
                except Exception as e:
                    logger.error(f"Error occurred while stopping task: {str(e)}")
                    return StopTaskResponse(stop_status="error", reason=str(e))
            else:
                return StopTaskResponse(stop_status="do_nothing", reason="No running task found.")

    def _setup_file_routes(self):
        @self.files_router.get("/download/{file_path:path}")
        async def download_file(file_path: str):
            """Download file"""
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
        @self.service_router.get("/status", response_model=ServiceStatusResponse)
        async def get_service_status():
            """Get service status"""
            return ServiceStatus.get_status_service()

        @self.service_router.get("/metadata", response_model=dict)
        async def get_service_metadata():
            """Get service metadata"""
            assert self.inference_service is not None, "Inference service is not initialized"
            return self.inference_service.server_metadata()

    def _process_video_generation(self, message: TaskRequest, stop_event: threading.Event):
        assert self.video_service is not None, "Video service is not initialized"
        try:
            if stop_event.is_set():
                logger.info(f"Task {message.task_id} received stop signal, terminating")
                ServiceStatus.record_failed_task(message, error="Task stopped")
                return

            # Use video generation service to process task
            result = asyncio.run(self.video_service.generate_video(message))

        except Exception as e:
            logger.error(f"Task {message.task_id} processing failed: {str(e)}")
            ServiceStatus.record_failed_task(message, error=str(e))

    def initialize_services(self, cache_dir: Path, inference_service: DistributedInferenceService):
        self.file_service = FileService(cache_dir)
        self.inference_service = inference_service
        self.video_service = VideoGenerationService(self.file_service, inference_service)

    def get_app(self) -> FastAPI:
        return self.app
