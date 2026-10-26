"""Released StreamRender-H3 3D RoPE frequencies for native H3 blocks."""

import torch


class StreamRenderH3RoPE:
    def __init__(self, inv_freq):
        if inv_freq.shape != (16,):
            raise ValueError(f"Expected 16 StreamRender-H3 RoPE frequencies, got {tuple(inv_freq.shape)}")
        self.inv_freq = inv_freq.to(torch.float32)

    def embeddings(self, position_ids, *, dtype=torch.bfloat16):
        """Return cos/sin in the source's BF16-rounded split-half layout."""
        if position_ids.ndim != 2 or position_ids.shape[1] != 3:
            raise ValueError(f"Expected [tokens, 3] H3 positions, got {tuple(position_ids.shape)}")
        positions = position_ids.to(torch.float32)
        phases = positions.unsqueeze(-1) * self.inv_freq.to(positions.device).view(1, 1, -1)
        temporal, height, width = phases.unbind(dim=1)
        half = torch.cat((temporal, height, width), dim=-1)
        cosine = half.cos().to(dtype)
        sine = half.sin().to(dtype)
        return torch.cat((cosine, cosine), dim=-1), torch.cat((sine, sine), dim=-1)
