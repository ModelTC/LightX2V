import torch
import torch.nn.functional as F


class QwenImagePostInfer:
    def __init__(self, config):
        self.config = config
        self.cpu_offload = config.get("cpu_offload", False)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, hidden_states, temb):
        temb1 = F.silu(temb)
        temb1 = weights.norm_out_linear.apply(temb1)
        scale, shift = torch.chunk(temb1, 2, dim=1)
        hidden_states = weights.norm_out.apply(hidden_states) * (1 + scale) + shift
        output = weights.proj_out_linear.apply(hidden_states.squeeze(0))
        return output.unsqueeze(0)
