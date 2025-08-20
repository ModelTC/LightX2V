# Code source: https://github.com/RiseAI-Sys/ParaVAE/blob/main/paravae/dist/distributed_env.py

import torch.distributed as dist
from torch.distributed import ProcessGroup
import os

class DistributedEnv:
    _vae_group = None
    _local_rank = None
    _world_size = None

    @classmethod
    def initialize(cls, vae_group: ProcessGroup):
        if vae_group is None:
            cls._vae_group = dist.group.WORLD
        else:
            cls._vae_group = vae_group
        cls._local_rank = int(os.environ.get('LOCAL_RANK', 0)) # FIXME: in ray all local_rank is 0
        cls._rank_mapping = None
        cls._init_rank_mapping()

    @classmethod
    def get_vae_group(cls) -> ProcessGroup:
        if cls._vae_group is None:
            raise RuntimeError("DistributedEnv not initialized. Call initialize() first.")
        return cls._vae_group

    @classmethod
    def get_global_rank(cls) -> int:
        return dist.get_rank()

    @classmethod
    def _init_rank_mapping(cls):
        """Initialize the mapping between group ranks and global ranks"""
        if cls._rank_mapping is None:
            # Get all ranks in the group
            ranks = [None] * cls.get_group_world_size()
            dist.all_gather_object(ranks, cls.get_global_rank(), group=cls.get_vae_group())
            cls._rank_mapping = ranks

    @classmethod
    def get_global_rank_from_group_rank(cls, group_rank: int) -> int:
        """Convert a rank in VAE group to global rank using cached mapping.

        Args:
            group_rank: The rank in VAE group

        Returns:
            The corresponding global rank

        Raises:
            RuntimeError: If the group_rank is invalid
        """
        if cls._rank_mapping is None:
            cls._init_rank_mapping()

        if group_rank < 0 or group_rank >= cls.get_group_world_size():
            raise RuntimeError(f"Invalid group rank: {group_rank}. Must be in range [0, {cls.get_group_world_size()-1}]")

        return cls._rank_mapping[group_rank]

    @classmethod
    def get_rank_in_vae_group(cls) -> int:
        return dist.get_rank(cls.get_vae_group())

    @classmethod
    def get_group_world_size(cls) -> int:
        return dist.get_world_size(cls.get_vae_group())

    @classmethod
    def get_local_rank(cls) -> int:
        return cls._local_rank
