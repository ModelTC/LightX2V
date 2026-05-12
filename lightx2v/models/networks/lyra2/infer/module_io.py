"""
Dataclasses that carry intermediate tensors between pre / transformer / post
inference stages of Lyra2WanDiT.

Lyra2 source reference:
  lyra_2/_src/networks/wan2pt1_lyra2.py  Lyra2WanModel.forward  (L806-1018)
"""
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Lyra2PreInferOutput:
    # Patchified token sequence  [B, L, dim]
    x_tokens: torch.Tensor
    # Per-token RoPE frequencies  [L, 1, 1, head_dim]
    freqs_tokens: torch.Tensor
    # Encoded text context         [B, L_ctx, dim]
    context: torch.Tensor
    # Per-sample AdaLN 6-vector    [B, 6, dim]
    e0_B_6_D: torch.Tensor
    # Time embedding before projection  [B, dim]
    e_B_D: torch.Tensor
    # Optional Plücker camera tokens  [B, L, cam_dim] or None
    camera_tokens: Optional[torch.Tensor]
    # Optional correspondence buffer tokens  [B, L, dim] or None
    buffer_tokens: Optional[torch.Tensor]
    # Token range of the generation chunk  (gen_start, gen_end)
    gen_range: tuple
    # Spatial grid of the generation chunk  (f_gen, h_gen, w_gen)
    gen_grid: tuple
