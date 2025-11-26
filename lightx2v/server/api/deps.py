from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
from loguru import logger

from ..services import DistributedInferenceService, FileService, ImageGenerationService, VideoGenerationService


class ServiceContainer:
    _instance: Optional["ServiceContainer"] = None

    def __init__(self):
        self.file_service: Optional[FileService] = None
        self.inference_service: Optional[DistributedInferenceService] = None
        self.video_service: Optional[VideoGenerationService] = None
        self.image_service: Optional[ImageGenerationService] = None
        self.max_queue_size: int = 10

    @classmethod
    def get_instance(cls) -> "ServiceContainer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self, cache_dir: Path, inference_service: DistributedInferenceService, max_queue_size: int = 10):
        self.file_service = FileService(cache_dir)
        self.inference_service = inference_service
        self.video_service = VideoGenerationService(self.file_service, inference_service)
        self.image_service = ImageGenerationService(self.file_service, inference_service)
        self.max_queue_size = max_queue_size


def get_services() -> ServiceContainer:
    return ServiceContainer.get_instance()


async def validate_url_async(url: str) -> bool:
    if not url or not url.startswith("http"):
        return True

    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return False

        timeout = httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)
        async with httpx.AsyncClient(verify=False, timeout=timeout) as client:
            response = await client.head(url, follow_redirects=True)
            return response.status_code < 400
    except Exception as e:
        logger.warning(f"URL validation failed for {url}: {str(e)}")
        return False
