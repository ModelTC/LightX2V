from dataclasses import dataclass

import torch

from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3PreInferOutput


@dataclass
class MiniMaxH3CausalPreInferOutput(MiniMaxH3PreInferOutput):
    key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None
    valid_sequence_length: int | None = None
