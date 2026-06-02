import torch

from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer


class WanS2VPostInfer(WanPostInfer):
    @torch.no_grad()
    def infer(self, x, pre_infer_out):
        if x.dim() == 3:
            x = x.squeeze(0)
        # Truncation to video tokens happens before head (see infer_non_blocks).
        return super().infer(x, pre_infer_out)
