import torch


def patchify(hidden_states: torch.Tensor, patch_size: int = 2, f_patch_size: int = 1) -> torch.Tensor:
    B, C, H, W = hidden_states.shape
    pH = pW = patch_size
    pF = f_patch_size
    F = 1
    F_tokens = F // pF
    H_tokens = H // pH
    W_tokens = W // pW

    hidden_states = hidden_states.view(B, C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
    hidden_states = hidden_states.permute(0, 2, 4, 6, 3, 5, 7, 1)
    hidden_states = hidden_states.reshape(B, F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

    return hidden_states
