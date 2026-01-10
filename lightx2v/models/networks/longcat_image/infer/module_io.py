from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LongCatImagePreInferModuleOutput:
    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor
    temb: torch.Tensor
    image_rotary_emb: tuple  # (cos, sin) tuple
    # For I2I (image editing) task
    input_image_latents: Optional[torch.Tensor] = None
    output_seq_len: Optional[int] = None
