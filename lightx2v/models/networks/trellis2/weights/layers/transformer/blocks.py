# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

import torch
import torch.nn as nn


class AbsolutePositionEmbedderWeights(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """

    _inference_type = ("lightx2v.models.networks.trellis2.infer.layers.transformer.blocks", "AbsolutePositionEmbedderInfer")

    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / 10000**self.freqs

    def forward(self, *args, **kwargs):
        from lightx2v.models.networks.trellis2.infer.layers.transformer.blocks import AbsolutePositionEmbedderInfer

        return AbsolutePositionEmbedderInfer(self).infer(*args, **kwargs)


class FeedForwardNetWeights(nn.Module):
    _inference_type = ("lightx2v.models.networks.trellis2.infer.layers.transformer.blocks", "FeedForwardNetInfer")

    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, int(channels * mlp_ratio)), nn.GELU(approximate="tanh"), nn.Linear(int(channels * mlp_ratio), channels))

    def forward(self, *args, **kwargs):
        from lightx2v.models.networks.trellis2.infer.layers.transformer.blocks import FeedForwardNetInfer

        return FeedForwardNetInfer(self).infer(*args, **kwargs)
