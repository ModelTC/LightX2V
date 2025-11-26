from typing import Optional

from loguru import logger

from .worker import TorchrunInferenceWorker


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
        if self.worker and self.worker.rank != 0:
            await self.worker.worker_loop()
