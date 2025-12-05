from typing import Any, Optional

from ..file_service import FileService
from ..inference import DistributedInferenceService
from .base import BaseGenerationService


class VideoGenerationService(BaseGenerationService):
    def __init__(self, file_service: FileService, inference_service: DistributedInferenceService):
        super().__init__(file_service, inference_service)

    def get_output_extension(self) -> str:
        return ".mp4"

    def get_task_type(self) -> str:
        return "t2v,i2v,s2v"

    async def generate_with_stop_event(self, message: Any, stop_event) -> Optional[Any]:
        return await super().generate_with_stop_event(message, stop_event)

    async def generate_video_with_stop_event(self, message: Any, stop_event) -> Optional[Any]:
        return await self.generate_with_stop_event(message, stop_event)
