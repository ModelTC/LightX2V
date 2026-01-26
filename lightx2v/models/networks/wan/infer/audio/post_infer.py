import torch

from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.utils.envs import *
from loguru import logger

class WanAudioPostInfer(WanPostInfer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    @torch.no_grad()
    def infer(self, x, pre_infer_out):
        _, h, w = pre_infer_out.grid_sizes.tuple

        grid_sizes = (pre_infer_out.valid_latent_num, h, w)
        x = x[:pre_infer_out.valid_token_len]

        x = self.unpatchify(x, grid_sizes)
        if self.clean_cuda_cache:
            torch.cuda.empty_cache()

        return [u.float() for u in x]
