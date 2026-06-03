from dataclasses import dataclass

import torch


@dataclass
class HidreamPreInferOutput:
    inputs_embeds: torch.Tensor
    position_ids: torch.Tensor
    token_types: torch.Tensor
    vinput_mask: torch.Tensor
    visual_pos_masks: torch.Tensor | None = None
    deepstack_visual_embeds: list[torch.Tensor] | None = None
    cond_image_embeds: torch.Tensor | None = None
    cond_deepstack_image_embeds: list[torch.Tensor] | None = None
    tgt_image_len: int | None = None


@dataclass
class HidreamTransformerInferOutput:
    hidden_states: torch.Tensor
    vinput_mask: torch.Tensor
    tgt_image_len: int | None = None
