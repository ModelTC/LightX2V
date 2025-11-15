import torch

from lightx2v.utils.envs import *


class HunyuanVideo15PostInfer:
    def __init__(self, config):
        self.config = config
        self.unpatchify_channels = config["out_channels"]
        self.patch_size = config["patch_size"]  # (1, 1, 1)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @torch.no_grad()
    def infer(self, x, pre_infer_out):
        x = self.unpatchify(x, pre_infer_out.grid_sizes[0], pre_infer_out.grid_sizes[1], pre_infer_out.grid_sizes[2])
        return x

    def unpatchify(self, x, t, h, w):
        """
        Unpatchify a tensorized input back to frame format.

        Args:
            x (Tensor): Input tensor of shape (N, T, patch_size**2 * C)
            t (int): Number of time steps
            h (int): Height in patch units
            w (int): Width in patch units

        Returns:
            Tensor: Output tensor of shape (N, C, t * pt, h * ph, w * pw)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs
