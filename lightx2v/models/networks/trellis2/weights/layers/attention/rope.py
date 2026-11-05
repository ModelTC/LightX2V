# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

from typing import Tuple

import torch
import torch.nn as nn


class RotaryPositionEmbedderWeights(nn.Module):
    _inference_type = ("lightx2v.models.networks.trellis2.infer.layers.attention.rope", "RotaryPositionEmbedderInfer")

    def __init__(self, head_dim: int, dim: int = 3, rope_freq: Tuple[float, float] = (1.0, 10000.0)):
        super().__init__()
        assert head_dim % 2 == 0, "Head dim must be divisible by 2"
        self.head_dim = head_dim
        self.dim = dim
        self.rope_freq = rope_freq
        self.freq_dim = head_dim // 2 // dim
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32, device="cpu") / self.freq_dim
        self.freqs = rope_freq[0] / rope_freq[1] ** self.freqs

    def forward(self, *args, **kwargs):
        from lightx2v.models.networks.trellis2.infer.layers.attention.rope import RotaryPositionEmbedderInfer

        return RotaryPositionEmbedderInfer(self).infer(*args, **kwargs)
