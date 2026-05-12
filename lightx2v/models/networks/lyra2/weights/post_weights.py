"""
Post-transformer weight holders for Lyra2WanDiT.

Covers the Head submodule (lyra_2/_src/networks/wan2pt1.py L533-565):

  # Original Head.__init__:
  # self.norm       = WanLayerNorm(dim, eps)        – elementwise_affine=False → NO params
  # self.head       = nn.Linear(dim, prod(patch_size) * out_dim)
  # self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

  # Original Head.forward (wan2pt1.py L556-565):
  # e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
  # x = self.head(self.norm(x) * (1 + e[1]) + e[0])
"""

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, TENSOR_REGISTER


class Lyra2PostWeights(WeightModule):
    """
    Weight holder for the Head module.

    head.norm       – WanLayerNorm with elementwise_affine=False → no params (pure norm)
    head.head       – nn.Linear(dim, prod(patch_size)*out_dim)
    head.modulation – nn.Parameter(1, 2, dim)
    """

    def __init__(self, config: dict):
        super().__init__()

        # ------------------------------------------------------------------
        # head.modulation  – raw Parameter(1, 2, dim)
        # Original: self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        # State dict key: "head.modulation"  (no .weight suffix)
        # ------------------------------------------------------------------
        self.add_module("head_modulation", TENSOR_REGISTER["Default"]("head.modulation"))

        # ------------------------------------------------------------------
        # head.head  – Linear(dim, prod(patch_size)*out_dim)
        # Original: self.head = nn.Linear(dim, out_dim)  where out_dim already = prod*c
        # State dict key: "head.head.weight", "head.head.bias"
        # ------------------------------------------------------------------
        self.add_module(
            "head_linear",
            MM_WEIGHT_REGISTER["Default"]("head.head.weight", "head.head.bias"),
        )

        # head.norm has no learnable parameters (elementwise_affine=False).
        # The layer norm computation is handled in post_infer.py using torch.nn.functional.

    def to_cpu(self, non_blocking=True):
        for m in self._modules.values():
            if hasattr(m, "to_cpu"):
                m.to_cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=True):
        for m in self._modules.values():
            if hasattr(m, "to_cuda"):
                m.to_cuda(non_blocking=non_blocking)
