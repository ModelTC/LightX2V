import math

import torch
from torch import nn

from .neighbor_map import cpu_hash_neighbors


def _triple(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,) * 3


class SparseConv3d(nn.Module):
    """Stride-one submanifold convolution, with [out, kx, ky, kz, in] weights."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None, neighbor_map=cpu_hash_neighbors):
        super().__init__()
        if _triple(stride) != (1, 1, 1) or padding not in (None, 0, (0, 0, 0)):
            raise ValueError("SparseConv3d supports stride=1 and padding=0 only")
        self.kernel_size = _triple(kernel_size)
        self.dilation = _triple(dilation)
        self.out_channels = out_channels
        self.neighbor_map = neighbor_map
        self.weight = nn.Parameter(torch.empty(out_channels, *self.kernel_size, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(in_channels * math.prod(self.kernel_size))
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        features = x.feats.to(self.weight.dtype)
        output = features.new_zeros((features.shape[0], self.out_channels))
        for (kx, ky, kz), indices in self.neighbor_map(x.coords.to(features.device), self.kernel_size, self.dilation):
            valid = indices >= 0
            if not valid.any():
                continue
            indices = indices.clamp_min(0).to(features.device)
            gathered = features.index_select(0, indices)
            contribution = gathered @ self.weight[:, kx, ky, kz, :].transpose(0, 1)
            contribution.mul_(valid.to(device=features.device, dtype=contribution.dtype).unsqueeze(1))
            output.add_(contribution)
        if self.bias is not None:
            output.add_(self.bias)
        return x.replace(output)
