from dataclasses import dataclass

import torch


@dataclass
class HidreamPreInferOutput:
    inputs_embeds: torch.Tensor
    rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    idx_ar: torch.Tensor
    idx_gen: torch.Tensor
    vinput_mask: torch.Tensor
    visual_pos_masks: torch.Tensor | None = None
    deepstack_visual_embeds: list[torch.Tensor] | None = None
    cond_image_embeds: torch.Tensor | None = None
    cond_deepstack_image_embeds: list[torch.Tensor] | None = None
    tgt_image_len: int | None = None
    inputs_embeds_ar: torch.Tensor | None = None
    inputs_embeds_gen: torch.Tensor | None = None
    rope_cos_sin_ar: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    rope_cos_sin_gen: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    vinput_mask_gen: torch.Tensor | None = None
    seq_p_padding_size: int = 0


@dataclass
class HidreamTransformerInferOutput:
    hidden_states: torch.Tensor
    vinput_mask: torch.Tensor
    tgt_image_len: int | None = None
    seq_p_padding_size: int = 0
