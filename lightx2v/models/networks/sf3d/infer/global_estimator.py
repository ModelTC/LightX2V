# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from __future__ import annotations

from lightx2v.models.networks.sf3d.weights.global_estimator import Any, Float, Tensor, get_activation


def MultiHeadEstimator_forward(self, triplane: Float[Tensor, "B 3 F Ht Wt"]) -> dict[str, Any]:
    x = self.layers(triplane.reshape(triplane.shape[0], -1, triplane.shape[-2], triplane.shape[-1]))
    if self.cfg.pool == "max":
        x = x.amax(dim=[-2, -1])
    elif self.cfg.pool == "mean":
        x = x.mean(dim=[-2, -1])
    else:
        raise NotImplementedError
    out = {("decoder_" if head.add_to_decoder_features else "") + head.name: get_activation(head.output_activation)(self.heads[head.name](x) + head.output_bias) for head in self.cfg.heads}
    for head in self.cfg.heads:
        if head.shape:
            head_name = ("decoder_" if head.add_to_decoder_features else "") + head.name
            out[head_name] = out[head_name].reshape(*head.shape)
    return out
