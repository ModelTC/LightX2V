# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.

from torch import Tensor, nn

from .bricks import get_activation_layer


class ModulateDiT(nn.Module):
    def __init__(self, feat_dim: int, factor: int, act_type: str = "silu"):
        super().__init__()
        self.act = get_activation_layer(act_type)()
        self.linear = nn.Linear(feat_dim, factor * feat_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        from lightx2v.models.networks.hymotion.infer.modulate_layers import ModulateDiT_forward

        return ModulateDiT_forward(self, x)


def modulate(x: Tensor, shift: Tensor | None = None, scale: Tensor | None = None) -> Tensor:
    if shift is not None and scale is not None:
        assert len(x.shape) == len(shift.shape) == len(scale.shape), (
            f"x, shift, scale must have the same number of dimensions, but got x.shape: {x.shape}, shift.shape: {shift.shape} and scale.shape: {scale.shape}"
        )
    if shift is not None and scale is not None:
        return x * (1 + scale) + shift
    elif shift is not None:
        return x + shift
    elif scale is not None:
        return x * (1 + scale)
    else:
        return x


def apply_gate(x: Tensor, gate: Tensor | None = None, tanh: bool = False) -> Tensor:
    if gate is not None:
        assert len(x.shape) == len(gate.shape), f"x, gate must have the same number of dimensions, but got {x.shape} and {gate.shape}"
    if gate is None:
        return x
    if tanh:
        return x * gate.tanh()
    else:
        return x * gate


__all__ = ["Tensor"]
