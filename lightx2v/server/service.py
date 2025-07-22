import queue
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
import torch.multiprocessing as mp
from loguru import logger

from ..utils.set_config import set_config
from ..infer import init_runner
from .utils import ServiceStatus
from .schema import TaskRequest, TaskResponse
from .distributed_utils import create_distributed_worker


mp.set_start_method("spawn", force=True)


class FileService:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.input_image_dir = cache_dir / "inputs" / "imgs"
        self.input_audio_dir = cache_dir / "inputs" / "audios"
        self.output_video_dir = cache_dir / "outputs"

        # Create directories
        for directory in [
            self.input_image_dir,
            self.output_video_dir,
            self.input_audio_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    async def download_image(self, image_url: str) -> Path:
        try:
            async with httpx.AsyncClient(verify=False) as client:
                response = await client.get(image_url)

            if response.status_code != 200:
                raise ValueError(f"Failed to download image from {image_url}")

            image_name = Path(urlparse(image_url).path).name
            if not image_name:
                raise ValueError(f"Invalid image URL: {image_url}")

            image_path = self.input_image_dir / image_name
            image_path.parent.mkdir(parents=True, exist_ok=True)

            with open(image_path, "wb") as f:
                f.write(response.content)

            return image_path
        except Exception as e:
            logger.error(f"Failed to download image: {e}")
            raise

    def save_uploaded_file(self, file_content: bytes, filename: str) -> Path:
        file_extension = Path(filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = self.input_image_dir / unique_filename

        with open(file_path, "wb") as f:
            f.write(file_content)

        return file_path

    def get_output_path(self, save_video_path: str) -> Path:
        video_path = Path(save_video_path)
        if not video_path.is_absolute():
            return self.output_video_dir / save_video_path
        return video_path


def _distributed_inference_worker(rank, world_size, master_addr, master_port, args, task_queue, result_queue):
    task_data = None
    worker = None

    try:
        logger.info(f"Process {rank}/{world_size - 1} initializing distributed inference service...")

        # Create and initialize distributed worker process
        worker = create_distributed_worker(rank, world_size, master_addr, master_port)
        if not worker.init():
            raise RuntimeError(f"Rank {rank} distributed environment initialization failed")

        # Initialize configuration and model
        config = set_config(args)
        logger.info(f"Rank {rank} config: {config}")

        runner = init_runner(config)
        logger.info(f"Process {rank}/{world_size - 1} distributed inference service initialization completed")

        while True:
            # Only rank=0 reads tasks from queue
            if rank == 0:
                try:
                    task_data = task_queue.get(timeout=1.0)
                    if task_data is None:  # Stop signal
                        logger.info(f"Process {rank} received stop signal, exiting inference service")
                        # Broadcast stop signal to other processes
                        worker.dist_manager.broadcast_task_data(None)
                        break
                    # Broadcast task data to other processes
                    worker.dist_manager.broadcast_task_data(task_data)
                except queue.Empty:
                    # Queue is empty, continue waiting
                    continue
            else:
                # Non-rank=0 processes receive task data from rank=0
                task_data = worker.dist_manager.broadcast_task_data()
                if task_data is None:  # Stop signal
                    logger.info(f"Process {rank} received stop signal, exiting inference service")
                    break

            # All processes handle the task
            if task_data is not None:
                logger.info(f"Process {rank} received inference task: {task_data['task_id']}")

                try:
                    # Set inputs and run inference
                    runner.set_inputs(task_data)  # type: ignore
                    runner.run_pipeline()

                    # Synchronize and report results
                    worker.sync_and_report(
                        task_data["task_id"],
                        "success",
                        result_queue,
                        save_video_path=task_data["save_video_path"],
                        message="Inference completed",
                    )
                except Exception as e:
                    logger.error(f"Process {rank} error occurred while processing task: {str(e)}")

                    # Synchronize and report error
                    worker.sync_and_report(
                        task_data.get("task_id", "unknown"),
                        "failed",
                        result_queue,
                        error=str(e),
                        message=f"Inference failed: {str(e)}",
                    )

    except KeyboardInterrupt:
        logger.info(f"Process {rank} received KeyboardInterrupt, gracefully exiting")
    except Exception as e:
        logger.error(f"Distributed inference service process {rank} startup failed: {str(e)}")
        if rank == 0:
            error_result = {
                "task_id": "startup",
                "status": "startup_failed",
                "error": str(e),
                "message": f"Inference service startup failed: {str(e)}",
            }
            result_queue.put(error_result)
    finally:
        try:
            if worker:
                worker.cleanup()
        except:  # noqa: E722
            pass


class DistributedInferenceService:
    def __init__(self):
        self.task_queue = None
        self.result_queue = None
        self.processes = []
        self.is_running = False

    def start_distributed_inference(self, args) -> bool:
        if hasattr(args, "lora_path") and args.lora_path:
            args.lora_configs = [{"path": args.lora_path, "strength": getattr(args, "lora_strength", 1.0)}]
            delattr(args, "lora_path")
            if hasattr(args, "lora_strength"):
                delattr(args, "lora_strength")

        self.args = args
        if self.is_running:
            logger.warning("Distributed inference service is already running")
            return True

        nproc_per_node = args.nproc_per_node
        if nproc_per_node <= 0:
            logger.error("nproc_per_node must be greater than 0")
            return False

        try:
            import random

            master_addr = "127.0.0.1"
            master_port = str(random.randint(20000, 29999))
            logger.info(f"Distributed inference service Master Addr: {master_addr}, Master Port: {master_port}")

            # Create shared queues
            self.task_queue = mp.Queue()
            self.result_queue = mp.Queue()

            # Start processes
            for rank in range(nproc_per_node):
                p = mp.Process(
                    target=_distributed_inference_worker,
                    args=(
                        rank,
                        nproc_per_node,
                        master_addr,
                        master_port,
                        args,
                        self.task_queue,
                        self.result_queue,
                    ),
                    daemon=True,
                )
                p.start()
                self.processes.append(p)

            self.is_running = True
            logger.info(f"Distributed inference service started successfully with {nproc_per_node} processes")
            return True

        except Exception as e:
            logger.exception(f"Error occurred while starting distributed inference service: {str(e)}")
            self.stop_distributed_inference()
            return False

    def stop_distributed_inference(self):
        if not self.is_running:
            return

        try:
            logger.info(f"Stopping {len(self.processes)} distributed inference service processes...")

            # Send stop signal
            if self.task_queue:
                for _ in self.processes:
                    self.task_queue.put(None)

            # Wait for processes to end
            for p in self.processes:
                try:
                    p.join(timeout=10)
                    if p.is_alive():
                        logger.warning(f"Process {p.pid} did not end within the specified time, forcing termination...")
                        p.terminate()
                        p.join(timeout=5)
                except:  # noqa: E722
                    pass

            logger.info("All distributed inference service processes have stopped")

        except Exception as e:
            logger.error(f"Error occurred while stopping distributed inference service: {str(e)}")
        finally:
            # Clean up resources
            self._clean_queues()
            self.processes = []
            self.task_queue = None
            self.result_queue = None
            self.is_running = False

    def _clean_queues(self):
        for queue_obj in [self.task_queue, self.result_queue]:
            if queue_obj:
                try:
                    while not queue_obj.empty():
                        queue_obj.get_nowait()
                except:  # noqa: E722
                    pass

    def submit_task(self, task_data: dict) -> bool:
        if not self.is_running or not self.task_queue:
            logger.error("Distributed inference service is not started")
            return False

        try:
            self.task_queue.put(task_data)
            return True
        except Exception as e:
            logger.error(f"Failed to submit task: {str(e)}")
            return False

    def wait_for_result(self, task_id: str, timeout: int = 300) -> Optional[dict]:
        if not self.is_running or not self.result_queue:
            return None

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                result = self.result_queue.get(timeout=1.0)

                if result.get("task_id") == task_id:
                    return result
                else:
                    # Not the result for current task, put back in queue
                    self.result_queue.put(result)
                    time.sleep(0.1)

            except queue.Empty:
                continue

        return None

    def server_metadata(self):
        assert hasattr(self, "args"), "Distributed inference service has not been started. Call start_distributed_inference() first."
        return {"nproc_per_node": self.args.nproc_per_node, "model_cls": self.args.model_cls, "model_path": self.args.model_path}


class VideoGenerationService:
    def __init__(self, file_service: FileService, inference_service: DistributedInferenceService):
        self.file_service = file_service
        self.inference_service = inference_service

    async def generate_video(self, message: TaskRequest) -> TaskResponse:
        try:
            task_data = {field: getattr(message, field) for field in message.model_fields_set if field != "task_id"}
            task_data["task_id"] = message.task_id

            if "image_path" in message.model_fields_set and message.image_path.startswith("http"):
                image_path = await self.file_service.download_image(message.image_path)
                task_data["image_path"] = str(image_path)

            save_video_path = self.file_service.get_output_path(message.save_video_path)
            task_data["save_video_path"] = str(save_video_path)

            if not self.inference_service.submit_task(task_data):
                raise RuntimeError("Distributed inference service is not started")

            result = self.inference_service.wait_for_result(message.task_id)

            if result is None:
                raise RuntimeError("Task processing timeout")

            if result.get("status") == "success":
                ServiceStatus.complete_task(message)
                return TaskResponse(
                    task_id=message.task_id,
                    task_status="completed",
                    save_video_path=str(save_video_path),
                )
            else:
                error_msg = result.get("error", "Inference failed")
                ServiceStatus.record_failed_task(message, error=error_msg)
                raise RuntimeError(error_msg)

        except Exception as e:
            logger.error(f"Task {message.task_id} processing failed: {str(e)}")
            ServiceStatus.record_failed_task(message, error=str(e))
            raise
