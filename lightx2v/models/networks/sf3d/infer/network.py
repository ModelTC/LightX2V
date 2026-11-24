# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from __future__ import annotations

from lightx2v.models.networks.sf3d.weights.network import Float, List, Optional, Tensor, conditional_decorator, custom_bwd, custom_fwd, get_activation, get_device, rearrange, torch


def PixelShuffleUpsampleNetwork_forward(self, triplanes: Float[Tensor, "B 3 Ci Hp Wp"]) -> Float[Tensor, "B 3 Co Hp2 Wp2"]:
    return rearrange(self.upsample(rearrange(triplanes, "B Np Ci Hp Wp -> (B Np) Ci Hp Wp", Np=3)), "(B Np) Co Hp Wp -> B Np Co Hp Wp", Np=3)


@conditional_decorator(custom_fwd, "cuda" in get_device(), cast_inputs=torch.float32, device_type="cuda")
def _TruncExp_forward(ctx, x):
    ctx.save_for_backward(x)
    return torch.exp(x)


@conditional_decorator(custom_bwd, "cuda" in get_device())
def _TruncExp_backward(ctx, g):
    x = ctx.saved_tensors[0]
    return g * torch.exp(torch.clamp(x, max=15))


def MaterialMLP_forward(self, x, include: Optional[List] = None, exclude: Optional[List] = None):
    if include is not None and exclude is not None:
        raise ValueError("Cannot specify both include and exclude.")
    if include is not None:
        heads = [h for h in self.cfg.heads if h.name in include]
    elif exclude is not None:
        heads = [h for h in self.cfg.heads if h.name not in exclude]
    else:
        heads = self.cfg.heads
    out = {head.name: get_activation(head.output_activation)(self.heads[head.name](x) + head.out_bias) for head in heads}
    return out
