# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

import torch.nn.functional as F

from ..mesh_extraction import flexible_dual_grid_to_mesh
from .sparse_unet_vae import SparseUnetVaeDecoder


class FlexiDualGridVaeDecoder(SparseUnetVaeDecoder):
    def __init__(self, resolution, model_channels, latent_channels, num_blocks, block_type, up_block_type, block_args, voxel_margin=0.5):
        super().__init__(7, model_channels, latent_channels, num_blocks, block_type, up_block_type, block_args)
        self.resolution = resolution
        self.voxel_margin = voxel_margin

    def forward(self, x, return_subs=False):
        decoded = super().forward(x, return_subs=return_subs)
        (values, subs) = decoded if return_subs else (decoded, None)
        values = values.float()
        vertices = values.replace((1 + 2 * self.voxel_margin) * F.sigmoid(values.feats[..., :3]) - self.voxel_margin)
        flags = values.replace(values.feats[..., 3:6] > 0)
        weights = values.replace(F.softplus(values.feats[..., 6:7]))
        meshes = [flexible_dual_grid_to_mesh(v.coords[:, 1:], v.feats, i.feats, w.feats, aabb=[[-0.5] * 3, [0.5] * 3], grid_size=self.resolution) for (v, i, w) in zip(vertices, flags, weights)]
        return (meshes, subs) if return_subs else meshes
