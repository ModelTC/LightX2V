import json
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image
from loguru import logger

from ...media import is_base64_audio, is_base64_image, save_base64_audio, save_base64_image
from ...schema import TaskResponse
from ..file_service import FileService
from ..inference import DistributedInferenceService


class BaseGenerationService(ABC):
    def __init__(self, file_service: FileService, inference_service: DistributedInferenceService):
        self.file_service = file_service
        self.inference_service = inference_service

    @abstractmethod
    def get_output_extension(self) -> str:
        pass

    @abstractmethod
    def get_task_type(self) -> str:
        pass

    def _is_target_task_type(self) -> bool:
        if self.inference_service.worker and self.inference_service.worker.runner:
            task_type = self.inference_service.worker.runner.config.get("task", "t2v")
            return task_type in self.get_task_type().split(",")
        return False

    async def _resolve_image_path(self, image_path: str) -> str:
        if not image_path:
            return ""

        if image_path.startswith("http"):
            downloaded_path = await self.file_service.download_image(image_path)
            return str(downloaded_path)
        elif is_base64_image(image_path):
            saved_path = save_base64_image(image_path, str(self.file_service.input_image_dir))
            return str(saved_path)
        else:
            return image_path

    async def _process_image_path(self, image_path: str, task_data: Dict[str, Any]) -> None:
        task_data["image_path"] = await self._resolve_image_path(image_path)

    async def _process_image_mask_path(self, image_mask_path: str, task_data: Dict[str, Any]) -> None:
        if not image_mask_path:
            return

        task_data["image_mask_path"] = await self._resolve_image_path(image_mask_path)

    def _pack_image_and_mask_as_dir(self, task_data: Dict[str, Any]) -> None:
        image_path = task_data.get("image_path", "")
        image_mask_path = task_data.get("image_mask_path", "")
        if not image_path or not image_mask_path:
            return

        image_file = Path(image_path)
        mask_file = Path(image_mask_path)
        if not image_file.exists() or not image_file.is_file():
            raise RuntimeError(f"Invalid image_path for mask mode: {image_path}")
        if not mask_file.exists() or not mask_file.is_file():
            raise RuntimeError(f"Invalid image_mask_path for mask mode: {image_mask_path}")

        image_pair_dir = self.file_service.input_image_dir / f"mask_pair_{uuid.uuid4().hex[:8]}"
        image_pair_dir.mkdir(parents=True, exist_ok=True)

        base_name = image_file.stem or "image"
        image_dst = image_pair_dir / f"{base_name}.png"
        mask_dst = image_pair_dir / f"{base_name}_mask.png"
        with Image.open(image_file) as image_obj:
            image_rgb = image_obj.convert("RGB")
            target_size = image_rgb.size
            image_rgb.save(image_dst)

        with Image.open(mask_file) as mask_obj:
            mask_rgb = mask_obj.convert("RGB")
            if mask_rgb.size != target_size:
                # Keep mask aligned with the reference image size for edit-mode latent shapes.
                mask_rgb = mask_rgb.resize(target_size, Image.NEAREST)
            mask_rgb.save(mask_dst)

        task_data["image_path"] = str(image_pair_dir)
        task_data["image_mask_path"] = ""

    async def _process_audio_path(self, audio_path: str, task_data: Dict[str, Any]) -> None:
        if not audio_path:
            return

        if audio_path.startswith("http"):
            downloaded_path = await self.file_service.download_audio(audio_path)
            task_data["audio_path"] = str(downloaded_path)
        elif is_base64_audio(audio_path):
            saved_path = save_base64_audio(audio_path, str(self.file_service.input_audio_dir))
            task_data["audio_path"] = str(saved_path)
        else:
            task_data["audio_path"] = audio_path

    async def _process_talk_objects(self, talk_objects: list, task_data: Dict[str, Any]) -> None:
        if not talk_objects:
            return

        task_data["talk_objects"] = [{} for _ in range(len(talk_objects))]

        for index, talk_object in enumerate(talk_objects):
            if talk_object.audio.startswith("http"):
                audio_path = await self.file_service.download_audio(talk_object.audio)
                task_data["talk_objects"][index]["audio"] = str(audio_path)
            elif is_base64_audio(talk_object.audio):
                audio_path = save_base64_audio(talk_object.audio, str(self.file_service.input_audio_dir))
                task_data["talk_objects"][index]["audio"] = str(audio_path)
            else:
                task_data["talk_objects"][index]["audio"] = talk_object.audio

            if talk_object.mask.startswith("http"):
                mask_path = await self.file_service.download_image(talk_object.mask)
                task_data["talk_objects"][index]["mask"] = str(mask_path)
            elif is_base64_image(talk_object.mask):
                mask_path = save_base64_image(talk_object.mask, str(self.file_service.input_image_dir))
                task_data["talk_objects"][index]["mask"] = str(mask_path)
            else:
                task_data["talk_objects"][index]["mask"] = talk_object.mask

        temp_path = self.file_service.cache_dir / uuid.uuid4().hex[:8]
        temp_path.mkdir(parents=True, exist_ok=True)
        task_data["audio_path"] = str(temp_path)

        config_path = temp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump({"talk_objects": task_data["talk_objects"]}, f)

    def _prepare_output_path(self, save_result_path: str, task_data: Dict[str, Any]) -> None:
        actual_save_path = self.file_service.get_output_path(save_result_path)
        if not actual_save_path.suffix:
            actual_save_path = actual_save_path.with_suffix(self.get_output_extension())
        task_data["save_result_path"] = str(actual_save_path)

    async def generate_with_stop_event(self, message: Any, stop_event) -> Optional[Any]:
        try:
            task_data = {field: getattr(message, field) for field in message.model_fields_set if field != "task_id"}
            task_data["task_id"] = message.task_id
            task_data["target_shape"] = message.target_shape

            if stop_event.is_set():
                logger.info(f"Task {message.task_id} cancelled before processing")
                return None

            if hasattr(message, "image_path") and message.image_path:
                await self._process_image_path(message.image_path, task_data)
                logger.info(f"Task {message.task_id} image path: {task_data.get('image_path')}")

            if hasattr(message, "image_mask_path") and message.image_mask_path:
                await self._process_image_mask_path(message.image_mask_path, task_data)
                logger.info(f"Task {message.task_id} image mask path: {task_data.get('image_mask_path')}")
                self._pack_image_and_mask_as_dir(task_data)
                logger.info(f"Task {message.task_id} packed image+mask dir: {task_data.get('image_path')}")

            if hasattr(message, "audio_path") and message.audio_path:
                await self._process_audio_path(message.audio_path, task_data)
                logger.info(f"Task {message.task_id} audio path: {task_data.get('audio_path')}")

            if hasattr(message, "talk_objects") and message.talk_objects:
                await self._process_talk_objects(message.talk_objects, task_data)

            self._prepare_output_path(message.save_result_path, task_data)
            task_data["seed"] = message.seed
            task_data["resize_mode"] = message.resize_mode

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
                error_type = result.get("error_type", "")
                exc = RuntimeError(error_msg)
                exc.original_error_type = error_type
                raise exc

        except Exception as e:
            logger.exception(f"Task {message.task_id} processing failed: {str(e)}")
            raise
