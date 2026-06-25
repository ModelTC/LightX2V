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
    seq_p_gen_len: int | None = None
    seq_p_gen_padding_size: int = 0
    seq_p_local_gen_len: int | None = None


@dataclass
class Cosmos3TransformerInferModuleOutput:
    und_seq: torch.Tensor
    gen_seq: torch.Tensor
