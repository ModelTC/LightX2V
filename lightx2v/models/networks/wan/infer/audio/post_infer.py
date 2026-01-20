import torch

from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.utils.envs import *


class WanAudioPostInfer(WanPostInfer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    @torch.no_grad()
    def infer(self, x, pre_infer_out):
        t, h, w = pre_infer_out.grid_sizes.tuple
        if self.config.get("f2v_process", False):
            grid_sizes = (t, h, w)
        else:
            grid_sizes = (t - 1, h, w)

        x = self.unpatchify(x, grid_sizes)

        if self.clean_cuda_cache:
            torch.cuda.empty_cache()

        return [u.float() for u in x]
