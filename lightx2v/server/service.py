import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx
import torch
from loguru import logger

from ..infer import init_runner
from ..utils.set_config import set_config
from .audio_utils import is_base64_audio, save_base64_audio
from .distributed_utils import DistributedManager
from .image_utils import is_base64_image, save_base64_image
from .schema import TaskRequest, TaskResponse


class FileService:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.input_image_dir = cache_dir / "inputs" / "imgs"
        self.input_audio_dir = cache_dir / "inputs" / "audios"
        self.output_video_dir = cache_dir / "outputs"

        self._http_client = None
        self._client_lock = asyncio.Lock()

        self.max_retries = 3
        self.retry_delay = 1.0
        self.max_retry_delay = 10.0

        for directory in [
            self.input_image_dir,
            self.output_video_dir,
            self.input_audio_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create a persistent HTTP client with connection pooling."""
        async with self._client_lock:
            if self._http_client is None or self._http_client.is_closed:
                timeout = httpx.Timeout(
                    connect=10.0,
                    read=30.0,
                    write=10.0,
                    pool=5.0,
                )
                limits = httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=30.0)
                self._http_client = httpx.AsyncClient(verify=False, timeout=timeout, limits=limits, follow_redirects=True)
            return self._http_client

    async def _download_with_retry(self, url: str, max_retries: Optional[int] = None) -> httpx.Response:
        """Download with exponential backoff retry logic."""
        if max_retries is None:
            max_retries = self.max_retries

        last_exception = None

        retry_delay = self.retry_delay

        for attempt in range(max_retries):
            try:
                client = await self._get_http_client()
                response = await client.get(url)

                if response.status_code == 200:
                    return response
                elif response.status_code >= 500:
                    logger.warning(f"Server error {response.status_code} for {url}, attempt {attempt + 1}/{max_retries}")
                    last_exception = httpx.HTTPStatusError(f"Server returned {response.status_code}", request=response.request, response=response)
                else:
                    raise httpx.HTTPStatusError(f"Client error {response.status_code}", request=response.request, response=response)

            except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as e:
                logger.warning(f"Connection error for {url}, attempt {attempt + 1}/{max_retries}: {str(e)}")
                last_exception = e
            except httpx.HTTPStatusError as e:
                if e.response and e.response.status_code < 500:
                    raise
                last_exception = e
            except Exception as e:
                logger.error(f"Unexpected error downloading {url}: {str(e)}")
                last_exception = e

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

        error_msg = f"All {max_retries} connection attempts failed for {url}"
        if last_exception:
            error_msg += f": {str(last_exception)}"
        raise httpx.ConnectError(error_msg)

    async def download_image(self, image_url: str) -> Path:
        """Download image with retry logic and proper error handling."""
        try:
            parsed_url = urlparse(image_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL format: {image_url}")

            response = await self._download_with_retry(image_url)

            image_name = Path(parsed_url.path).name
            if not image_name:
                image_name = f"{uuid.uuid4()}.jpg"

            image_path = self.input_image_dir / image_name
            image_path.parent.mkdir(parents=True, exist_ok=True)

            with open(image_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Successfully downloaded image from {image_url} to {image_path}")
            return image_path

        except httpx.ConnectError as e:
            logger.error(f"Connection error downloading image from {image_url}: {str(e)}")
            raise ValueError(f"Failed to connect to {image_url}: {str(e)}")
        except httpx.TimeoutException as e:
            logger.error(f"Timeout downloading image from {image_url}: {str(e)}")
            raise ValueError(f"Download timeout for {image_url}: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading image from {image_url}: {str(e)}")
            raise ValueError(f"HTTP error for {image_url}: {str(e)}")
        except ValueError as e:
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading image from {image_url}: {str(e)}")
            raise ValueError(f"Failed to download image from {image_url}: {str(e)}")

    async def download_audio(self, audio_url: str) -> Path:
        """Download audio with retry logic and proper error handling."""
        try:
            parsed_url = urlparse(audio_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL format: {audio_url}")

            response = await self._download_with_retry(audio_url)

            audio_name = Path(parsed_url.path).name
            if not audio_name:
                audio_name = f"{uuid.uuid4()}.mp3"

            audio_path = self.input_audio_dir / audio_name
            audio_path.parent.mkdir(parents=True, exist_ok=True)

            with open(audio_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Successfully downloaded audio from {audio_url} to {audio_path}")
            return audio_path

        except httpx.ConnectError as e:
            logger.error(f"Connection error downloading audio from {audio_url}: {str(e)}")
            raise ValueError(f"Failed to connect to {audio_url}: {str(e)}")
        except httpx.TimeoutException as e:
            logger.error(f"Timeout downloading audio from {audio_url}: {str(e)}")
            raise ValueError(f"Download timeout for {audio_url}: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading audio from {audio_url}: {str(e)}")
            raise ValueError(f"HTTP error for {audio_url}: {str(e)}")
        except ValueError as e:
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading audio from {audio_url}: {str(e)}")
            raise ValueError(f"Failed to download audio from {audio_url}: {str(e)}")

    def save_uploaded_file(self, file_content: bytes, filename: str) -> Path:
        file_extension = Path(filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = self.input_image_dir / unique_filename

        with open(file_path, "wb") as f:
            f.write(file_content)

        return file_path

    def get_output_path(self, save_result_path: str) -> Path:
        video_path = Path(save_result_path)
        if not video_path.is_absolute():
            return self.output_video_dir / save_result_path
        return video_path

    async def cleanup(self):
        """Cleanup resources including HTTP client."""
        async with self._client_lock:
            if self._http_client and not self._http_client.is_closed:
                await self._http_client.aclose()
                self._http_client = None


class TorchrunInferenceWorker:
    """Worker class for torchrun-based distributed inference"""

    def __init__(self):
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.runner = None
        self.dist_manager = DistributedManager()
        self.processing = False  # Track if currently processing a request

    def init(self, args) -> bool:
        """Initialize the worker with model and distributed setup"""
        try:
            # Initialize distributed process group using torchrun env vars
            if self.world_size > 1:
                if not self.dist_manager.init_process_group():
                    raise RuntimeError("Failed to initialize distributed process group")
            else:
                # Single GPU mode
                self.dist_manager.rank = 0
                self.dist_manager.world_size = 1
                self.dist_manager.device = "cuda:0" if torch.cuda.is_available() else "cpu"
                self.dist_manager.is_initialized = False

            # Initialize model
            config = set_config(args)
            if self.rank == 0:
                logger.info(f"Config:\n {json.dumps(config, ensure_ascii=False, indent=4)}")

            self.runner = init_runner(config)
            logger.info(f"Rank {self.rank}/{self.world_size - 1} initialization completed")

            return True

        except Exception as e:
            logger.error(f"Rank {self.rank} initialization failed: {str(e)}")
            return False

    async def process_request(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single inference request

        Note: We keep the inference synchronous to maintain NCCL/CUDA context integrity.
        The async wrapper allows FastAPI to handle other requests while this runs.
        """
        try:
            # Only rank 0 broadcasts task data (worker processes already received it in worker_loop)
            if self.world_size > 1 and self.rank == 0:
                task_data = self.dist_manager.broadcast_task_data(task_data)

            # Run inference directly - torchrun handles the parallelization
            # Using asyncio.to_thread would be risky with NCCL operations
            # Instead, we rely on FastAPI's async handling and queue management
            self.runner.set_inputs(task_data)
            self.runner.run_pipeline()

            # Small yield to allow other async operations if needed
            await asyncio.sleep(0)

            # Synchronize all ranks
            if self.world_size > 1:
                self.dist_manager.barrier()

            # Only rank 0 returns the result
            if self.rank == 0:
                return {
                    "task_id": task_data["task_id"],
                    "status": "success",
                    "save_result_path": task_data.get("video_path", task_data["save_result_path"]),
                    "message": "Inference completed",
                }
            else:
                return None

        except Exception as e:
            logger.error(f"Rank {self.rank} inference failed: {str(e)}")
            if self.world_size > 1:
                self.dist_manager.barrier()

            if self.rank == 0:
                return {
                    "task_id": task_data.get("task_id", "unknown"),
                    "status": "failed",
                    "error": str(e),
                    "message": f"Inference failed: {str(e)}",
                }
            else:
                return None

    async def worker_loop(self):
        """Non-rank-0 workers: Listen for broadcast tasks"""
        while True:
            try:
                task_data = self.dist_manager.broadcast_task_data()
                if task_data is None:
                    logger.info(f"Rank {self.rank} received stop signal")
                    break

                await self.process_request(task_data)

            except Exception as e:
                logger.error(f"Rank {self.rank} worker loop error: {str(e)}")
                continue

    def cleanup(self):
        self.dist_manager.cleanup()


class DistributedInferenceService:
    def __init__(self):
        self.worker = None
        self.is_running = False
        self.args = None

    def start_distributed_inference(self, args) -> bool:
        self.args = args
        if self.is_running:
            logger.warning("Distributed inference service is already running")
            return True

        try:
            self.worker = TorchrunInferenceWorker()

            if not self.worker.init(args):
                raise RuntimeError("Worker initialization failed")

            self.is_running = True
            logger.info(f"Rank {self.worker.rank} inference service started successfully")
            return True

        except Exception as e:
            logger.error(f"Error starting inference service: {str(e)}")
            self.stop_distributed_inference()
            return False

    def stop_distributed_inference(self):
        if not self.is_running:
            return

        try:
            if self.worker:
                self.worker.cleanup()
            logger.info("Inference service stopped")
        except Exception as e:
            logger.error(f"Error stopping inference service: {str(e)}")
        finally:
            self.worker = None
            self.is_running = False

    async def submit_task_async(self, task_data: dict) -> Optional[dict]:
        if not self.is_running or not self.worker:
            logger.error("Inference service is not started")
            return None

        if self.worker.rank != 0:
            return None

        try:
            if self.worker.processing:
                # If we want to support queueing, we can add the task to queue
                # For now, we'll process sequentially
                logger.info(f"Waiting for previous task to complete before processing task {task_data.get('task_id')}")

            self.worker.processing = True
            result = await self.worker.process_request(task_data)
            self.worker.processing = False
            return result
        except Exception as e:
            self.worker.processing = False
            logger.error(f"Failed to process task: {str(e)}")
            return {
                "task_id": task_data.get("task_id", "unknown"),
                "status": "failed",
                "error": str(e),
                "message": f"Task processing failed: {str(e)}",
            }

    def server_metadata(self):
        assert hasattr(self, "args"), "Distributed inference service has not been started. Call start_distributed_inference() first."
        return {"nproc_per_node": self.worker.world_size, "model_cls": self.args.model_cls, "model_path": self.args.model_path}

    async def run_worker_loop(self):
        """Run the worker loop for non-rank-0 processes"""
        if self.worker and self.worker.rank != 0:
            await self.worker.worker_loop()


class VideoGenerationService:
    def __init__(self, file_service: FileService, inference_service: DistributedInferenceService):
        self.file_service = file_service
        self.inference_service = inference_service

    async def generate_video_with_stop_event(self, message: TaskRequest, stop_event) -> Optional[TaskResponse]:
        """Generate video using torchrun-based inference"""
        try:
            task_data = {field: getattr(message, field) for field in message.model_fields_set if field != "task_id"}
            task_data["task_id"] = message.task_id

            if stop_event.is_set():
                logger.info(f"Task {message.task_id} cancelled before processing")
                return None

            if "image_path" in message.model_fields_set and message.image_path:
                if message.image_path.startswith("http"):
                    image_path = await self.file_service.download_image(message.image_path)
                    task_data["image_path"] = str(image_path)
                elif is_base64_image(message.image_path):
                    image_path = save_base64_image(message.image_path, str(self.file_service.input_image_dir))
                    task_data["image_path"] = str(image_path)
                else:
                    task_data["image_path"] = message.image_path

                logger.info(f"Task {message.task_id} image path: {task_data['image_path']}")

            if "audio_path" in message.model_fields_set and message.audio_path:
                if message.audio_path.startswith("http"):
                    audio_path = await self.file_service.download_audio(message.audio_path)
                    task_data["audio_path"] = str(audio_path)
                elif is_base64_audio(message.audio_path):
                    audio_path = save_base64_audio(message.audio_path, str(self.file_service.input_audio_dir))
                    task_data["audio_path"] = str(audio_path)
                else:
                    task_data["audio_path"] = message.audio_path

                logger.info(f"Task {message.task_id} audio path: {task_data['audio_path']}")

            actual_save_path = self.file_service.get_output_path(message.save_result_path)
            task_data["save_result_path"] = str(actual_save_path)
            task_data["video_path"] = message.save_result_path

            result = await self.inference_service.submit_task_async(task_data)

            if result is None:
                if stop_event.is_set():
                    logger.info(f"Task {message.task_id} cancelled during processing")
                    return None
                raise RuntimeError("Task processing failed")

            if result.get("status") == "success":
                return TaskResponse(
                    task_id=message.task_id,
                    task_status="completed",
                    save_result_path=message.save_result_path,  # Return original path
                )
            else:
                error_msg = result.get("error", "Inference failed")
                raise RuntimeError(error_msg)

        except Exception as e:
            logger.error(f"Task {message.task_id} processing failed: {str(e)}")
            raise
