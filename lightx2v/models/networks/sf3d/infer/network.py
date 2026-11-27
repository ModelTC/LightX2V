# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from __future__ import annotations

from lightx2v.models.networks.sf3d.weights.network import Float, List, Optional, Tensor, get_activation, rearrange, torch


def PixelShuffleUpsampleNetwork_forward(self, triplanes: Float[Tensor, "B 3 Ci Hp Wp"]) -> Float[Tensor, "B 3 Co Hp2 Wp2"]:
    return rearrange(self.upsample(rearrange(triplanes, "B Np Ci Hp Wp -> (B Np) Ci Hp Wp", Np=3)), "(B Np) Co Hp Wp -> B Np Co Hp Wp", Np=3)


def _TruncExp_forward(ctx, x):
    return torch.exp(x)


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
