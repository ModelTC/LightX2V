"""Deterministic data cursor for local Turbo experiments."""

import math

import torch
from torch.utils.data import Sampler


class TurboCursorSampler(Sampler):
    def __init__(self, dataset, *, seed=42, shuffle=True, num_replicas=1, rank=0):
        if not len(dataset):
            raise ValueError("Turbo requires a non-empty dataset")
        self.dataset = dataset
        self.num_replicas, self.rank = int(num_replicas), int(rank)
        if self.num_replicas < 1 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("Turbo data cursor requires a positive world size and a valid rank")
        self.seed, self.shuffle = int(seed), bool(shuffle)
        self.epoch, self.start_microbatch = 0, 0

    def configure_from_microbatch_offset(self, *, start_microbatch, **kwargs):
        del kwargs
        self.start_microbatch = int(start_microbatch)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.num_replicas)

    def __iter__(self):
        start = self.start_microbatch + self.epoch * len(self)
        permutations = {}
        size = len(self.dataset)
        for microbatch in range(start, start + len(self)):
            epoch, offset = divmod(microbatch * self.num_replicas + self.rank, size)
            if epoch not in permutations:
                generator = torch.Generator().manual_seed(self.seed + epoch)
                permutations[epoch] = torch.randperm(size, generator=generator).tolist() if self.shuffle else list(range(size))
            yield permutations[epoch][offset]

    def checkpoint_metadata(self, **kwargs):
        del kwargs
        return {"sampler": "turbo_global_shuffle_cursor_v1", "seed": self.seed, "shuffle": self.shuffle, "num_samples": len(self.dataset), "data_parallel_world_size": self.num_replicas}
