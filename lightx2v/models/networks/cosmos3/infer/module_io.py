from dataclasses import dataclass

import torch


@dataclass
class Cosmos3PreInferModuleOutput:
    hidden_states: torch.Tensor
    und_len: int
    position_ids: torch.Tensor
    vision_mse_loss_indexes: torch.Tensor
    vision_token_shapes: list
    vision_noisy_frame_indexes: list
    original_latent_shapes: list
    sound_mse_loss_indexes: torch.Tensor | None = None
    sound_token_shapes: list | None = None
    sound_noisy_frame_indexes: list | None = None
    action_mse_loss_indexes: torch.Tensor | None = None
    action_token_shapes: list | None = None
    action_noisy_frame_indexes: list | None = None
    action_domain_ids: torch.Tensor | None = None
    raw_action_dim: int | None = None
    seq_p_gen_len: int | None = None
    seq_p_gen_padding_size: int = 0
    seq_p_local_gen_len: int | None = None


@dataclass
class Cosmos3TransformerInferModuleOutput:
    und_seq: torch.Tensor
    gen_seq: torch.Tensor


@dataclass
class Cosmos3PostInferModuleOutput:
    vision: torch.Tensor
    sound: torch.Tensor | None = None
    action: torch.Tensor | None = None
