# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from __future__ import annotations

from lightx2v.models.networks.sf3d.weights.camera import torch


def LinearCameraEmbedder_forward(self, **kwargs):
    cond_tensors = []
    for cond_name in self.cfg.conditions:
        assert cond_name in kwargs
        cond = kwargs[cond_name]
        cond_tensors.append(cond.view(*cond.shape[:2], -1))
    cond_tensor = torch.cat(cond_tensors, dim=-1)
    assert cond_tensor.shape[-1] == self.cfg.in_channels
    embedding = self.linear(cond_tensor)
    return embedding
