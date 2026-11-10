from .base import BaseGenerationService


class VideoGenerationService(BaseGenerationService):
    def get_output_extension(self) -> str:
        return ".mp4"
