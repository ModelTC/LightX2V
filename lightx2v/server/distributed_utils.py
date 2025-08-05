import os

import torch
import torch.distributed as dist
from loguru import logger


class DistributedManager:
    def __init__(self):
        self.is_initialized = False
        self.rank = 0
        self.world_size = 1

    def init_process_group(self, rank: int, world_size: int, master_addr: str, master_port: str) -> bool:
        try:
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port

            dist.init_process_group(backend="nccl", init_method=f"tcp://{master_addr}:{master_port}", rank=rank, world_size=world_size)

            if torch.cuda.is_available():  # type: ignore
                torch.cuda.set_device(rank)

            self.is_initialized = True
            self.rank = rank
            self.world_size = world_size

            logger.info(f"Rank {rank}/{world_size - 1} distributed environment initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Rank {rank} distributed environment initialization failed: {str(e)}")
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

    def barrier(self):
        if self.is_initialized:
            dist.barrier()

    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def broadcast_task_data(self, task_data=None):  # type: ignore
        if not self.is_initialized:
            return None

        if self.is_rank_zero():
            if task_data is None:
                stop_signal = torch.tensor([1], dtype=torch.int32, device=f"cuda:{self.rank}")
            else:
                stop_signal = torch.tensor([0], dtype=torch.int32, device=f"cuda:{self.rank}")

            dist.broadcast(stop_signal, src=0)

            if task_data is not None:
                import pickle

                task_bytes = pickle.dumps(task_data)
                task_length = torch.tensor([len(task_bytes)], dtype=torch.int32, device=f"cuda:{self.rank}")

                dist.broadcast(task_length, src=0)

                task_tensor = torch.tensor(list(task_bytes), dtype=torch.uint8, device=f"cuda:{self.rank}")
                dist.broadcast(task_tensor, src=0)

                return task_data
            else:
                return None
        else:
            stop_signal = torch.tensor([0], dtype=torch.int32, device=f"cuda:{self.rank}")
            dist.broadcast(stop_signal, src=0)

            if stop_signal.item() == 1:
                return None
            else:
                task_length = torch.tensor([0], dtype=torch.int32, device=f"cuda:{self.rank}")
                dist.broadcast(task_length, src=0)

                task_tensor = torch.empty(int(task_length.item()), dtype=torch.uint8, device=f"cuda:{self.rank}")
                dist.broadcast(task_tensor, src=0)

                import pickle

                task_bytes = bytes(task_tensor.cpu().numpy())
                task_data = pickle.loads(task_bytes)

                return task_data


class DistributedWorker:
    def __init__(self, rank: int, world_size: int, master_addr: str, master_port: str):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.dist_manager = DistributedManager()

    def init(self) -> bool:
        return self.dist_manager.init_process_group(self.rank, self.world_size, self.master_addr, self.master_port)

    def cleanup(self):
        self.dist_manager.cleanup()

    def sync_and_report(self, task_id: str, status: str, result_queue, **kwargs):
        # Synchronize all processes
        self.dist_manager.barrier()

        if self.dist_manager.is_rank_zero():
            result = {"task_id": task_id, "status": status, **kwargs}
            result_queue.put(result)
            logger.info(f"Task {task_id} {status}")


def create_distributed_worker(rank: int, world_size: int, master_addr: str, master_port: str) -> DistributedWorker:
    return DistributedWorker(rank, world_size, master_addr, master_port)
