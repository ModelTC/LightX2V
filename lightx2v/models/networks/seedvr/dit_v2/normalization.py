

from typing import Callable, Optional
from diffusers.models.normalization import RMSNorm
from torch import nn

# (dim: int, eps: float, elementwise_affine: bool)
norm_layer_type = Callable[[int, float, bool], nn.Module]


def get_norm_layer(norm_type: Optional[str]) -> norm_layer_type:

    def _norm_layer(dim: int, eps: float, elementwise_affine: bool):
        if norm_type is None:
            return nn.Identity()

        if norm_type == "layer":
            return nn.LayerNorm(
                normalized_shape=dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

        if norm_type == "rms":
            return RMSNorm(
                dim=dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

        if norm_type == "fusedln":
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(
                normalized_shape=dim,
                elementwise_affine=elementwise_affine,
                eps=eps,
            )

        if norm_type == "fusedrms":
            from apex.normalization import FusedRMSNorm

            return FusedRMSNorm(
                normalized_shape=dim,
                elementwise_affine=elementwise_affine,
                eps=eps,
            )

        raise NotImplementedError(f"{norm_type} is not supported")

    return _norm_layer
