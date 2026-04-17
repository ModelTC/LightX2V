from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Flux2PreInferModuleOutput:
    """Output from Flux2 pre-inference stage.

    Mirrors Flux2 pipeline preprocessing:
    - Embedded image latents via x_embedder
    - Embedded text embeddings via context_embedder
    - Timestep embeddings via timestep_embedder MLP
    - Position IDs for text and image (txt_ids, img_ids)
    - Rotary embeddings for attention
    - Optional I2I conditioning
    """

    hidden_states: torch.Tensor  # x_embedder output [L, D]
    encoder_hidden_states: torch.Tensor  # context_embedder output [L, D]
    timestep: torch.Tensor  # timestep_embedder MLP output [D]

    txt_ids: Optional[torch.Tensor] = None  # [B, text_seq_len, 4]
    img_ids: Optional[torch.Tensor] = None  # [B, image_seq_len, 4]

    image_rotary_emb: Optional[tuple] = None  # (cos, sin) tuple

    input_image_latents: Optional[torch.Tensor] = None
    output_seq_len: Optional[int] = None


# Backward-compatible alias
Flux2KleinPreInferModuleOutput = Flux2PreInferModuleOutput
