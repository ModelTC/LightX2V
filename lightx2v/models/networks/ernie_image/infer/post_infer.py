import torch.nn.functional as F


class ErnieImagePostInfer:
    def __init__(self, config):
        self.config = config
        self.patch_size = config.get("patch_size", 1)
        self.out_channels = config.get("out_channels", config.get("in_channels", 128))
        self.eps = config.get("eps", 1e-6)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, hidden_states, pre_infer_out):
        scale, shift = weights.final_norm_linear.apply(pre_infer_out.conditioning).chunk(2, dim=-1)
        hidden_states = F.layer_norm(
            hidden_states,
            (hidden_states.shape[-1],),
            weight=None,
            bias=None,
            eps=self.eps,
        )
        hidden_states = hidden_states * (1 + scale) + shift
        patches = weights.final_linear.apply(hidden_states)[: pre_infer_out.image_tokens_len]

        height, width = pre_infer_out.image_hw
        p = self.patch_size
        return patches.view(1, height, width, p, p, self.out_channels).permute(0, 5, 1, 3, 2, 4).contiguous().view(1, self.out_channels, height * p, width * p)
