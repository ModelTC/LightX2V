from pathlib import Path
from typing import Any, Optional

from loguru import logger

from ...schema import ImageTaskRequest, TaskResponse
from .base import BaseGenerationService


class ImageGenerationService(BaseGenerationService):
    def get_output_extension(self) -> str:
        return ".png"

    def get_task_type(self) -> str:
        return "t2i,i2i"

    async def generate_with_stop_event(self, message: ImageTaskRequest, stop_event) -> Optional[Any]:
        try:
            task_data = self.prepare_task_data(message)

            if stop_event.is_set():
                logger.info(f"Task {message.task_id} cancelled before processing")
                return None

            if hasattr(message, "image_path") and message.image_path:
                task_data["image_path"] = await self._resolve_image_path(message.image_path)
                logger.info(f"Task {message.task_id} image path: {task_data.get('image_path')}")

            if hasattr(message, "image_mask_path") and message.image_mask_path:
                task_data["image_mask_path"] = await self._resolve_image_path(message.image_mask_path)
                logger.info(f"Task {message.task_id} image mask path: {task_data.get('image_mask_path')}")
                self._pack_image_and_mask_as_dir(task_data)
                logger.info(f"Task {message.task_id} packed image+mask dir: {task_data.get('image_path')}")

            task_data.pop("image_mask_path", None)
            prefer_memory_result = message._prefer_memory_result
            task_data.pop("presigned_url", None)
            if prefer_memory_result:
                task_data["return_result_tensor"] = True

            result = await self.inference_service.submit_task_async(task_data)

            if result is None:
                if stop_event.is_set():
                    logger.info(f"Task {message.task_id} cancelled during processing")
                    return None
                raise RuntimeError("Task processing failed")

            if result.get("status") == "success":
                if prefer_memory_result:
                    result_png = result.get("result_png")
                    if not result_png:
                        raise RuntimeError("Image inference did not return in-memory PNG bytes (result_png)")
                    usage = result.get("usage")
                    return TaskResponse(
                        task_id=message.task_id,
                        task_status="completed",
                        save_result_path=None,
                        result_png=result_png,
                        usage=usage,
                    )

                output_path = result["save_result_path"]
                return TaskResponse(
                    task_id=message.task_id,
                    task_status="completed",
                    save_result_path=str(Path(output_path).absolute()) if output_path is not None else None,
                )
            else:
                error_msg = result.get("error", "Inference failed")
                error_type = result.get("error_type", "")
                exc = RuntimeError(error_msg)
                exc.original_error_type = error_type
                raise exc

        except Exception as e:
            logger.exception(f"Task {message.task_id} processing failed: {str(e)}")
            raise
