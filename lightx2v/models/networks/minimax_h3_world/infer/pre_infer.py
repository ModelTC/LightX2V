"""Refine the scene prompt and each latent-frame action independently."""

import torch

from lightx2v.models.networks.minimax_h3.infer.pre_infer import MiniMaxH3PreInfer


class MiniMaxH3WorldPreInfer(MiniMaxH3PreInfer):
    def _refine_text(self, weights, text_embeds):
        segments = self.scheduler.layout.text_segment_lengths
        if len(segments) < 2 or any(length <= 0 for length in segments) or sum(segments) != text_embeds.shape[0]:
            raise ValueError("H3-World ia2av requires a scene prompt and one text segment per latent action")
        # Native dense wrappers ignore cu_seqlens for unbatched 3D inputs.
        # Separate calls preserve the prompt/action isolation during refinement.
        refine_segment = super()._refine_text
        return torch.cat([refine_segment(weights, part) for part in text_embeds.split(segments)], dim=0)
