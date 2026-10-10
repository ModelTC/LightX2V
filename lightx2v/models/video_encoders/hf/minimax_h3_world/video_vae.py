import torch

from lightx2v.models.video_encoders.hf.minimax_h3.video_vae import MiniMaxH3VideoVAE


class MiniMaxH3WorldVideoVAE(MiniMaxH3VideoVAE):
    """Use H3-World's deterministic posterior mean for image conditioning."""

    def _sample_condition_latents(self, moments: torch.Tensor) -> torch.Tensor:
        return self.normalize_latents(torch.chunk(moments, 2, dim=1)[0])
