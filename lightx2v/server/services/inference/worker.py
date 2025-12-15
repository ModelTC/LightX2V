import asyncio
import os
from typing import Any, Dict

import torch
from easydict import EasyDict
from loguru import logger

from lightx2v.infer import init_runner
from lightx2v.utils.input_info import set_input_info
from lightx2v.utils.set_config import set_config, set_parallel_config

from ..distributed_utils import DistributedManager


class TorchrunInferenceWorker:
    def __init__(self):
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.runner = None
        self.dist_manager = DistributedManager()
        self.processing = False

    def init(self, args) -> bool:
        try:
            if self.world_size > 1:
                if not self.dist_manager.init_process_group():
                    raise RuntimeError("Failed to initialize distributed process group")
            else:
                self.dist_manager.rank = 0
                self.dist_manager.world_size = 1
                self.dist_manager.device = "cuda:0" if torch.cuda.is_available() else "cpu"
                self.dist_manager.is_initialized = False

            config = set_config(args)

            if config["parallel"]:
                set_parallel_config(config)

            if self.rank == 0:
                logger.info(f"Config:\n {config}")

            self.runner = init_runner(config)
            logger.info(f"Rank {self.rank}/{self.world_size - 1} initialization completed")

            return True

        except Exception as e:
            logger.exception(f"Rank {self.rank} initialization failed: {str(e)}")
            return False

    async def process_request(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        has_error = False
        error_msg = ""

        try:
            if self.world_size > 1 and self.rank == 0:
                task_data = self.dist_manager.broadcast_task_data(task_data)

            task_data["task"] = self.runner.config["task"]
            task_data["return_result_tensor"] = False
            task_data["negative_prompt"] = task_data.get("negative_prompt", "")

            target_fps = task_data.pop("target_fps", None)
            if target_fps is not None:
                vfi_cfg = self.runner.config.get("video_frame_interpolation")
                if vfi_cfg:
                    task_data["video_frame_interpolation"] = {**vfi_cfg, "target_fps": target_fps}
                else:
                    logger.warning(f"Target FPS {target_fps} is set, but video frame interpolation is not configured")

            task_data = EasyDict(task_data)
            input_info = set_input_info(task_data)

            self.runner.set_config(task_data)
            self.runner.run_pipeline(input_info)

            await asyncio.sleep(0)

        except Exception as e:
            has_error = True
            error_msg = str(e)
            logger.exception(f"Rank {self.rank} inference failed: {error_msg}")

        if self.world_size > 1:
            self.dist_manager.barrier()

        if self.rank == 0:
            if has_error:
                return {
                    "task_id": task_data.get("task_id", "unknown"),
                    "status": "failed",
                    "error": error_msg,
                    "message": f"Inference failed: {error_msg}",
                }
            else:
                return {
                    "task_id": task_data["task_id"],
                    "status": "success",
                    "save_result_path": task_data.get("video_path", task_data["save_result_path"]),
                    "message": "Inference completed",
                }
        else:
            return None

    async def worker_loop(self):
        while True:
            task_data = None
            try:
                task_data = self.dist_manager.broadcast_task_data()
                if task_data is None:
                    logger.info(f"Rank {self.rank} received stop signal")
                    break

                await self.process_request(task_data)

            except Exception as e:
                error_str = str(e)
                if "Connection closed by peer" in error_str or "Connection reset by peer" in error_str:
                    logger.info(f"Rank {self.rank} detected master process shutdown, exiting worker loop")
                    break
                logger.error(f"Rank {self.rank} worker loop error: {error_str}")
                if self.world_size > 1 and task_data is not None:
                    try:
                        self.dist_manager.barrier()
                    except Exception as barrier_error:
                        logger.warning(f"Rank {self.rank} barrier failed, exiting: {barrier_error}")
                        break
                continue

    def cleanup(self):
        self.dist_manager.cleanup()
