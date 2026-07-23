from dataclasses import dataclass

import torch


@dataclass
class QwenPreInferModuleOutput:
    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor
    temb_img_silu: torch.Tensor
    temb_txt_silu: torch.Tensor
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor]
    image_rotary_positions: tuple[torch.Tensor | None, torch.Tensor | None]
