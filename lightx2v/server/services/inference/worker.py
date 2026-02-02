import asyncio
import os
from pathlib import Path
from typing import Any, Dict

import torch
from loguru import logger

from lightx2v.infer import init_runner
from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
from lightx2v.utils.set_config import set_config, set_parallel_config

from ..distributed_utils import DistributedManager


class TorchrunInferenceWorker:
    def __init__(self):
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.runner = None
        self.dist_manager = DistributedManager()
        self.processing = False
        self.lora_dir = None
        self.current_lora_name = None

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

            self.lora_dir = getattr(args, "lora_dir", None)
            if self.lora_dir:
                self.lora_dir = Path(self.lora_dir)
                if not self.lora_dir.exists():
                    logger.warning(f"LoRA directory does not exist: {self.lora_dir}")
                    self.lora_dir = None
                else:
                    logger.info(f"LoRA directory set to: {self.lora_dir}")

            config = set_config(args)

            if config["parallel"]:
                set_parallel_config(config)

            if self.rank == 0:
                logger.info(f"Config:\n {config}")

            self.runner = init_runner(config)
            logger.info(f"Rank {self.rank}/{self.world_size - 1} initialization completed")

            self.input_info = init_empty_input_info(args.task)

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

            # Handle dynamic LoRA loading
            lora_name = task_data.pop("lora_name", None)
            lora_strength = task_data.pop("lora_strength", 1.0)
            
            if self.lora_dir:
                self.switch_lora(lora_name, lora_strength)

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

            update_input_info_from_dict(self.input_info, task_data)

            self.runner.set_config(task_data)
            self.runner.run_pipeline(self.input_info)

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

    def switch_lora(self, lora_name: str, lora_strength: float):
        try:
            if lora_name is None:
                if self.current_lora_name is not None:
                    logger.info(f"Removing LoRA: {self.current_lora_name}")
                    if hasattr(self.runner.model, "remove_lora"):
                        self.runner.model.remove_lora()
                    self.current_lora_name = None
                return

            if lora_name != self.current_lora_name:
                lora_path = self._lora_path(lora_name)
                if lora_path is None:
                    logger.warning(f"LoRA file not found for: {lora_name}")
                    return

                logger.info(f"Loading LoRA: {lora_name} from {lora_path} with strength={lora_strength}")
                if hasattr(self.runner.model, "update_lora"):
                    self.runner.model.update_lora(lora_path, lora_strength)
                else:
                    logger.warning("Model does not support dynamic LoRA loading")
                    return
                self.current_lora_name = lora_name
                logger.info(f"LoRA loaded successfully: {lora_name}")

            elif lora_strength != getattr(self.runner.model, "lora_strength", 1.0):
                # Same LoRA but different strength - just update strength
                logger.info(f"Updating LoRA strength to {lora_strength}")
                if hasattr(self.runner.model, "lora_strength"):
                    self.runner.model.lora_strength = lora_strength

        except Exception as e:
            logger.error(f"Failed to handle LoRA switching: {e}")
            raise

    def _lora_path(self, lora_name: str) -> str:
        if not self.lora_dir:
            return None
        lora_file = self.lora_dir / lora_name
        if lora_file.exists():
            return str(lora_file)
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
