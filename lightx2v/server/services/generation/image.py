from typing import Any, Optional

from loguru import logger

from ...schema import TaskResponse
from ..file_service import FileService
from ..inference import DistributedInferenceService
from .base import BaseGenerationService


class ImageGenerationService(BaseGenerationService):
    def __init__(self, file_service: FileService, inference_service: DistributedInferenceService):
        super().__init__(file_service, inference_service)

    def get_output_extension(self) -> str:
        return ".png"

    def get_task_type(self) -> str:
        return "t2i,i2i"

    async def generate_with_stop_event(self, message: Any, stop_event) -> Optional[Any]:
        try:
            task_data = {field: getattr(message, field) for field in message.model_fields_set if field != "task_id"}
            task_data["task_id"] = message.task_id

            if hasattr(message, "aspect_ratio"):
                task_data["aspect_ratio"] = message.aspect_ratio

            if stop_event.is_set():
                logger.info(f"Task {message.task_id} cancelled before processing")
                return None

            if hasattr(message, "image_path") and message.image_path:
                await self._process_image_path(message.image_path, task_data)
                logger.info(f"Task {message.task_id} image path: {task_data.get('image_path')}")

            self._prepare_output_path(message.save_result_path, task_data)
            task_data["seed"] = message.seed

            result = await self.inference_service.submit_task_async(task_data)

            if result is None:
                if stop_event.is_set():
                    logger.info(f"Task {message.task_id} cancelled during processing")
                    return None
                raise RuntimeError("Task processing failed")

            if result.get("status") == "success":
                actual_save_path = self.file_service.get_output_path(message.save_result_path)
                if not actual_save_path.suffix:
                    actual_save_path = actual_save_path.with_suffix(self.get_output_extension())
                return TaskResponse(
                    task_id=message.task_id,
                    task_status="completed",
                    save_result_path=actual_save_path.name,
                )
            else:
                error_msg = result.get("error", "Inference failed")
                raise RuntimeError(error_msg)

        except Exception as e:
            logger.exception(f"Task {message.task_id} processing failed: {str(e)}")
            raise

    async def generate_image_with_stop_event(self, message: Any, stop_event) -> Optional[Any]:
        return await self.generate_with_stop_event(message, stop_event)
