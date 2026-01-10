import torch
import torch.nn.functional as F


class LongCatImagePostInfer:
    """Post-processing inference for LongCat Image Transformer."""

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
        # AdaLayerNormContinuous: linear -> chunk -> modulate
        # norm_out.linear projects temb to 2*D for scale and shift
        temb_out = F.linear(
            F.silu(temb),
            weights.norm_out_linear_weight,
            weights.norm_out_linear_bias
        )
        scale, shift = torch.chunk(temb_out, 2, dim=-1)

        # Layer norm (no learnable params) + modulation
        hidden_states = F.layer_norm(hidden_states, (hidden_states.shape[-1],))
        hidden_states = hidden_states * (1 + scale) + shift

        # Final projection
        output = F.linear(
            hidden_states,
            weights.proj_out_weight,
            weights.proj_out_bias
        )

        return output.unsqueeze(0)
