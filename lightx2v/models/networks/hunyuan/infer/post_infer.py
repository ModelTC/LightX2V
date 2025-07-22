import torch


class HunyuanPostInfer:
    def __init__(self, config):
        self.config = config

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, img, vec):
        out = torch.nn.functional.silu(vec)
        out = weights.final_layer_adaLN_modulation_1.apply(out)
        shift, scale = out.chunk(2, dim=1)
        out = torch.nn.functional.layer_norm(img, (img.shape[1],), None, None, 1e-6)
        out = out * (1 + scale) + shift
        out = weights.final_layer_linear.apply(out.to(torch.float32))
        _, _, ot, oh, ow = self.scheduler.latents.shape
        patch_size = [1, 2, 2]
        tt, th, tw = (
            ot // patch_size[0],
            oh // patch_size[1],
            ow // patch_size[2],
        )

        c = 16
        pt, ph, pw = patch_size

        out = out.reshape(shape=(1, tt, th, tw, c, pt, ph, pw))
        out = torch.einsum("nthwcopq->nctohpwq", out)
        out = out.reshape(shape=(1, c, tt * pt, th * ph, tw * pw))

        return out
