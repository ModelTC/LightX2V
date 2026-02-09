from dataclasses import dataclass

import torch

from lightx2v.models.networks.seedvr.dit_v2.cache import Cache


@dataclass
class SeedVRPreInferOutput:
    vid: torch.Tensor
    txt: torch.Tensor
    vid_shape: torch.LongTensor
    txt_shape: torch.LongTensor
    emb: torch.Tensor
    cache: Cache
