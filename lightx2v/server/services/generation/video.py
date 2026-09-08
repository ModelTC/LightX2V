from .base import BaseGenerationService


class VideoGenerationService(BaseGenerationService):
    def get_output_extension(self) -> str:
        return ".mp4"

    def get_task_type(self) -> str:
        return "t2v,i2v,s2v"
