from dataclasses import dataclass

import torch


@dataclass
class HunyuanImage3PreInferOutput:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    custom_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None
    past_key_values: object | None = None
    use_cache: bool = False
    image_mask: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    token_hw: tuple[int, int] | None = None
    first_step: bool | None = None
    full_attn_slices: list | None = None
