import torch.nn.functional as F


class QwenImage21PostInfer:
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, hidden, state):
        scale = weights.modulation.apply(F.silu(state.temb))
        return weights.proj.apply(weights.norm.apply(hidden) * (1 + scale))
