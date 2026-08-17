"""LTX-2.5-only weights and backend factories."""

import torch

from lightx2v.common.ops.rope.template import broadcast_freqs
from lightx2v.common.ops.rope.torch_rope import TorchRealRope
from lightx2v.models.networks.ltx2.weights.pre_weights import LTX2PreWeights
from lightx2v.utils.registry_factory import RMS_WEIGHT_REGISTER, ROPE_REGISTER, TENSOR_REGISTER


@ROPE_REGISTER("ltx25_split_rope")
class LTX25SplitRope(TorchRealRope):
    """Source-exact BF16 split-half RoPE for LTX-2.5."""

    def apply_single(
        self,
        x: torch.Tensor,
        freqs,
        rotary_dim: int | None = None,
        unsqueeze_dim: int = -2,
        **kwargs,
    ) -> torch.Tensor:
        rotary_dim = rotary_dim or x.shape[-1]
        if rotary_dim % 2:
            raise ValueError(f"rotary_dim must be even, got {rotary_dim}.")

        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        cos, sin, pairwise = self._cos_sin(freqs, rotary_dim)
        if self.layout != "split_half" or not pairwise:
            return super().apply_single(
                x,
                freqs,
                rotary_dim=rotary_dim,
                unsqueeze_dim=unsqueeze_dim,
                **kwargs,
            )

        first, second = x_rot.chunk(2, dim=-1)
        cos = broadcast_freqs(cos.to(dtype=x_rot.dtype), first, unsqueeze_dim)
        sin = broadcast_freqs(sin.to(dtype=x_rot.dtype), first, unsqueeze_dim)
        split_input = x_rot.reshape(*x_rot.shape[:-1], 2, rotary_dim // 2)
        output = split_input * cos.unsqueeze(-2)
        first_output = output[..., :1, :]
        second_output = output[..., 1:, :]
        first_output.addcmul_(-sin.unsqueeze(-2), split_input[..., 1:, :])
        second_output.addcmul_(sin.unsqueeze(-2), split_input[..., :1, :])
        output = output.flatten(-2)
        return torch.cat((output, x_pass), dim=-1) if x_pass.shape[-1] else output


class LTX25PreWeights(LTX2PreWeights):
    """LTX-2.5 pre-block weights, including the keyframe marker."""

    def __init__(self, config):
        super().__init__(config)
        if config.get("use_keyframes_abs_pos_embedding", False):
            self.add_module(
                "keyframes_abs_pos_embedding",
                TENSOR_REGISTER["Default"](
                    tensor_name="model.diffusion_model.keyframes_abs_pos_embedding",
                ),
            )


@RMS_WEIGHT_REGISTER("ltx25_fast")
def ltx25_fast_rms_weight(*args, **kwargs):
    """Use SGL for weighted Q/K norms and torch-native for block RMSNorm."""
    weight_name = kwargs.get("weight_name", args[0] if args else None)
    backend = "torch_native" if weight_name is None else "sgl-kernel"
    return RMS_WEIGHT_REGISTER[backend](*args, **kwargs)
