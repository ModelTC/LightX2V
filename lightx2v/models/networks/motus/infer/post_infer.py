import torch


class MotusPostInfer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.scheduler = None

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @torch.no_grad()
    def infer(self, action_latents: torch.Tensor, pre_infer_out):
        del pre_infer_out
        return self.model.denormalize_actions(action_latents.float()).squeeze(0)
