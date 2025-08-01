import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from loguru import logger

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


class SequenceParallelCoordinator:
    def __init__(self, config):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(dist.get_rank())
        self.config = config
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        if self.world_size == 1:
            self.cfg_parallel_size = 1
            self.seq_parallel_size = 1
            self.cfg_group = None
            self.seq_group = None
        else:
            if not is_power_of_two(self.world_size):
                raise Exception("parallel method acquires the world size in power of 2.")

            self.cfg_parallel_size = self.config.get("cfg_parallel_size", 2)
            self.seq_parallel_size = self.config.get("seq_parallel_size", None)
            if self.seq_parallel_size == None:
                self.seq_parallel_size = self.world_size // self.cfg_parallel_size
            
            self.mesh_2d = init_device_mesh("cuda", (self.cfg_parallel_size, self.seq_parallel_size), mesh_dim_names=("cfg", "seq"))
            # self.cfg_mesh = self.mesh_2d["cfg"]
            self.cfg_group = self.mesh_2d.get_group("cfg")
            # self.seq_mesh = self.mesh_2d["seq"]
            self.seq_group = self.mesh_2d.get_group("seq")
                
            if dist.get_rank() == 0 and self.mesh_2d is not None:
                logger.info(f"cfg group size: {dist.get_world_size(self.cfg_group)}")
                logger.info(f"seq group size: {dist.get_world_size(self.seq_group)}")

            dist.barrier()

            logger.info(f"CUDA:{self.rank} is set to cfg-group rank {dist.get_rank(self.cfg_group)}")

            dist.barrier()

            logger.info(f"CUDA:{self.rank} is set to seq-group rank {dist.get_rank(self.seq_group)}")
            
    def is_cfg_parallel(self):
        return self.cfg_parallel_size > 1
    
    def is_seq_parallel(self):
        return self.seq_parallel_size > 1
    
    def get_cfg_parallel_size(self):
        return self.cfg_parallel_size
    
    def get_seq_parallel_size(self):
        return self.seq_parallel_size

    def get_cfg_parallel_rank(self):
        return dist.get_rank(self.cfg_group)
    
    def get_seq_parallel_rank(self):
        return dist.get_rank(self.seq_group)
    
    def cfg_group(self):
        return self.cfg_group
    
    def seq_group(self):
        return self.seq_group
