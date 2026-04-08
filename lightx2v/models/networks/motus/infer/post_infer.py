import torch

from .module_io import MotusPostInferModuleOutput


class MotusPostInfer:
    def __init__(self, adapter, config):
        self.adapter = adapter
        self.config = config
        self.scheduler = None

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @torch.no_grad()
    def infer(self, video_latents: torch.Tensor, action_latents: torch.Tensor):
        decoded_frames = self.adapter.model.video_model.decode_video(video_latents)
        pred_frames = ((decoded_frames[:, :, 1:] + 1.0) / 2.0).clamp(0, 1).float()
        pred_actions = self.adapter.denormalize_actions(action_latents.float())
        return MotusPostInferModuleOutput(pred_frames=pred_frames, pred_actions=pred_actions)
