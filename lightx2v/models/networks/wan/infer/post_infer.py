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

    @torch.compile(disable=not CHECK_ENABLE_GRAPH_MODE())
    def infer(self, weights, x, e, grid_sizes):
        if e.dim() == 2:
            modulation = weights.head_modulation.tensor  # 1, 2, dim
            e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
        elif e.dim() == 3:  # For Diffustion forcing
            modulation = weights.head_modulation.tensor.unsqueeze(2)  # 1, 2, seq, dim
            e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
            e = [ei.squeeze(1) for ei in e]

        x = weights.norm.apply(x)

        if GET_DTYPE() != "BF16":
            x = x.float()
        x.mul_(1 + e[1].squeeze()).add_(e[0].squeeze())
        if GET_DTYPE() != "BF16":
            x = x.to(torch.bfloat16)

        x = weights.head.apply(x)
        x = self.unpatchify(x, grid_sizes)

        if self.clean_cuda_cache:
            del e, grid_sizes
            torch.cuda.empty_cache()

        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        x = x.unsqueeze(0)
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
