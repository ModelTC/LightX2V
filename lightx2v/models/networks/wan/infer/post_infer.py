import math

import torch

from lightx2v.utils.envs import *


class WanPostInfer:
    def __init__(self, config):
        self.out_dim = config["out_dim"]
        self.patch_size = (1, 2, 2)
        self.clean_cuda_cache = config.get("clean_cuda_cache", False)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, x, pre_infer_out):
        x = self.unpatchify(x, pre_infer_out.grid_sizes.tuple)

        if self.clean_cuda_cache:
            torch.cuda.empty_cache()

        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        x = x[: math.prod(grid_sizes)].view(*grid_sizes, *self.patch_size, c)
        x = torch.einsum("fhwpqrc->cfphqwr", x)
        x = x.reshape(c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return [x]
