import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from loguru import logger
from starlette.responses import RedirectResponse

from ..services import DistributedInferenceService
from ..task_manager import TaskStatus, task_manager
from .deps import ServiceContainer, get_services
from .router import create_api_router


class ApiServer:
    def __init__(self, max_queue_size: int = 10, app: Optional[FastAPI] = None):
        self.app = app or FastAPI(title="LightX2V API", version="1.0.0")
        self.max_queue_size = max_queue_size

        self.processing_thread = None
        self.stop_processing = threading.Event()

        self._setup_routes()

    def _setup_routes(self):
        @self.app.get("/")
        def redirect_to_docs():
            return RedirectResponse(url="/docs")

        api_router = create_api_router()
        self.app.include_router(api_router)

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
        services = get_services()

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

            from ..schema import ImageTaskRequest

            if isinstance(message, ImageTaskRequest):
                generation_service = services.image_service
            else:
                generation_service = services.video_service

            result = await generation_service.generate_with_stop_event(message, task_info.stop_event)

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
        container = ServiceContainer.get_instance()
        container.initialize(cache_dir, inference_service, self.max_queue_size)
        self._ensure_processing_thread_running()

    async def cleanup(self):
        self.stop_processing.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)

        services = get_services()
        if services.file_service:
            await services.file_service.cleanup()

    def get_app(self) -> FastAPI:
        return self.app
