from dataclasses import dataclass

import torch

from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3PreInferOutput


@dataclass(kw_only=True)
class MiniMaxH3CleanPreInferOutput(MiniMaxH3PreInferOutput):
    key_rotary_emb: tuple[torch.Tensor, torch.Tensor]
    valid_sequence_length: int
