import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler


class RelayVSRScheduler(BaseScheduler):
    """Official distilled flow step: sigma 1 -> 0, no classifier-free guidance."""

    def __init__(self, config):
        super().__init__(config)
        self.keep_latents_dtype_in_scheduler = True

    def prepare(self, seed, device):
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def prepare_latents(self, height, width, device):
        self.latents = torch.randn((1, 48, 1, height // 4, width // 4), generator=self.generator, device=device, dtype=torch.float32)

    def step_post(self):
        self.latents = self.latents - self.noise_pred.float()

    def clear(self):
        self.latents = self.noise_pred = self.generator = None
