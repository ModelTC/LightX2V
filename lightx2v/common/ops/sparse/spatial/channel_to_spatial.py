# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

import torch
from torch import nn

from ..basic import SparseTensor


class SparseChannel2Spatial(nn.Module):
    """Rearrange feature channels into the children selected by a subdivision mask."""

    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor

    def forward(self, x, subdivision):
        if subdivision is None:
            raise ValueError("Channel-to-spatial upsampling requires a subdivision mask")
        dim = x.coords.shape[-1] - 1
        mask = subdivision.feats
        children = mask.nonzero()[:, -1]
        counts = mask.sum(dim=-1)
        coords = x.coords.clone().detach()
        coords[:, 1:] *= self.factor
        coords = torch.repeat_interleave(coords, counts, dim=0, output_size=children.shape[0])
        for axis in range(dim):
            coords[:, axis + 1] += children // self.factor**axis % self.factor
        parents = torch.repeat_interleave(torch.arange(x.coords.shape[0], device=x.device), counts, dim=0, output_size=children.shape[0])
        features = x.feats.reshape(x.feats.shape[0] * self.factor**dim, -1)
        features = features[parents * self.factor**dim + children]
        shape = None if x._shape is None else torch.Size([x._shape[0], x._shape[1] // self.factor**dim])
        output = SparseTensor(features, coords, shape)
        output._scale = tuple(scale / self.factor for scale in x._scale)
        return output
