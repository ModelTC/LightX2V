import os
import pickle
from datetime import timedelta
from typing import Any, Optional

import torch
import torch.distributed as dist
from loguru import logger


class DistributedManager:
    def __init__(self):
        self.is_initialized = False
        self.rank = 0
        self.world_size = 1
        self.device = "cpu"
        self.task_pg = None

    CHUNK_SIZE = 1024 * 1024

    def init_process_group(self) -> bool:
        try:
            self.rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))

            if self.world_size > 1:
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                dist.init_process_group(backend=backend, init_method="env://")
                logger.info(f"Setup backend: {backend}")

                task_timeout = timedelta(days=30)
                self.task_pg = dist.new_group(backend="gloo", timeout=task_timeout)
                logger.info("Created gloo process group for task distribution with 30-day timeout")

                if torch.cuda.is_available():
                    torch.cuda.set_device(self.rank)
                    self.device = f"cuda:{self.rank}"
                else:
                    self.device = "cpu"
            else:
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

            self.is_initialized = True
            logger.info(f"Rank {self.rank}/{self.world_size - 1} distributed environment initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Rank {self.rank} distributed environment initialization failed: {str(e)}")
            return False

    def cleanup(self):
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
                logger.info(f"Rank {self.rank} distributed environment cleaned up")
        except Exception as e:
            logger.error(f"Rank {self.rank} error occurred while cleaning up distributed environment: {str(e)}")
        finally:
            self.is_initialized = False
            self.task_pg = None

    def barrier(self):
        if self.is_initialized:
            if torch.cuda.is_available() and dist.get_backend() == "nccl":
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()

    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def _broadcast_byte_chunks(self, data_bytes: bytes) -> None:
        total_length = len(data_bytes)
        num_full_chunks = total_length // self.CHUNK_SIZE
        remaining = total_length % self.CHUNK_SIZE

        for i in range(num_full_chunks):
            start_idx = i * self.CHUNK_SIZE
            end_idx = start_idx + self.CHUNK_SIZE
            chunk = data_bytes[start_idx:end_idx]
            task_tensor = torch.tensor(list(chunk), dtype=torch.uint8)
            dist.broadcast(task_tensor, src=0, group=self.task_pg)

        if remaining:
            chunk = data_bytes[-remaining:]
            task_tensor = torch.tensor(list(chunk), dtype=torch.uint8)
            dist.broadcast(task_tensor, src=0, group=self.task_pg)

    def _receive_byte_chunks(self, total_length: int) -> bytes:
        if total_length <= 0:
            return b""

        received = bytearray()
        remaining = total_length

        while remaining > 0:
            chunk_length = min(self.CHUNK_SIZE, remaining)
            task_tensor = torch.empty(chunk_length, dtype=torch.uint8)
            dist.broadcast(task_tensor, src=0, group=self.task_pg)
            received.extend(task_tensor.numpy())
            remaining -= chunk_length

        return bytes(received)

    def broadcast_task_data(self, task_data: Optional[Any] = None) -> Optional[Any]:
        if not self.is_initialized:
            return None

        if self.is_rank_zero():
            if task_data is None:
                stop_signal = torch.tensor([1], dtype=torch.int32)
            else:
                stop_signal = torch.tensor([0], dtype=torch.int32)

            dist.broadcast(stop_signal, src=0, group=self.task_pg)

            if task_data is not None:
                task_bytes = pickle.dumps(task_data)
                task_length = torch.tensor([len(task_bytes)], dtype=torch.int32)

                dist.broadcast(task_length, src=0, group=self.task_pg)
                self._broadcast_byte_chunks(task_bytes)

                return task_data
            else:
                return None
        else:
            stop_signal = torch.tensor([0], dtype=torch.int32)
            dist.broadcast(stop_signal, src=0, group=self.task_pg)

            if stop_signal.item() == 1:
                return None
            else:
                task_length = torch.tensor([0], dtype=torch.int32)

                dist.broadcast(task_length, src=0, group=self.task_pg)
                total_length = int(task_length.item())

                task_bytes = self._receive_byte_chunks(total_length)
                task_data = pickle.loads(task_bytes)
                return task_data
