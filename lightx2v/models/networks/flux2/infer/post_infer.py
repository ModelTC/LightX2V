import torch
import torch.nn.functional as F


class Flux2PostInfer:
    def __init__(self, config):
        self.config = config
        self.cpu_offload = config.get("cpu_offload", False)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, hidden_states, temb):
        """
        Run post-processing: final normalization and projection.

        Args:
            weights: Post-processing weights (norm_out, proj_out)
            hidden_states: Transformer output [L, D]
            temb: Timestep embedding [B, D]

        Returns:
            Output tensor [B, L, C]
        """
        temb_out = weights.norm_out_linear.apply(F.silu(temb))
        scale, shift = torch.chunk(temb_out, 2, dim=-1)

        hidden_states = F.layer_norm(hidden_states, (hidden_states.shape[-1],))
        hidden_states = hidden_states * (1 + scale) + shift

        output = weights.proj_out.apply(hidden_states)

        return output.unsqueeze(0)


# Backward-compatible alias
Flux2KleinPostInfer = Flux2PostInfer
