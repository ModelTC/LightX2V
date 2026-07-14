from dataclasses import dataclass

import torch


@dataclass
class ZPreInferModuleOutput:
    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor
    temb_img_silu: torch.Tensor
    temb_txt_silu: torch.Tensor
    x_freqs_cis: torch.Tensor
    cap_freqs_cis: torch.Tensor
    image_tokens_len: int
    x_item_seqlens: list
    cap_item_seqlens: list

    @property
    def adaln_input(self) -> torch.Tensor:
        return self.temb_img_silu

    @property
    def image_rotary_emb(self) -> torch.Tensor:
        return self.x_freqs_cis

    @property
    def freqs_cis(self) -> torch.Tensor:
        return self.x_freqs_cis
