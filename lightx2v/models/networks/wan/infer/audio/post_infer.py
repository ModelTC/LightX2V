import torch

from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.utils.envs import *


class WanAudioPostInfer(WanPostInfer):
    def __init__(self, config):
        super().__init__(config)

    @torch.compile(disable=not CHECK_ENABLE_GRAPH_MODE())
    def infer(self, x, pre_infer_out):
        x = x[: pre_infer_out.seq_lens[0]]

        t, h, w = pre_infer_out.grid_sizes.tuple
        grid_sizes = (t - 1, h, w)

        x = self.unpatchify(x, grid_sizes)

        if self.clean_cuda_cache:
            torch.cuda.empty_cache()

        return [u.float() for u in x]
