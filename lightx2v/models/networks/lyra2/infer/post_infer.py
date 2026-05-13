"""
Post-transformer inference for Lyra2WanDiT.

Implements Head.forward + unpatch rearrange (wan2pt1.py L556-565 and
Lyra2WanModel.forward L1003-1017).

  # Original Head.forward (wan2pt1.py L556-565):
  #   e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)   → scale/shift
  #   x = self.head(self.norm(x) * (1 + e[1]) + e[0])
  #
  # Original unpatch (wan2pt1_lyra2.py L1006-1017, framepack path):
  #   x_gen = x_B_L_D[:, gen_start:gen_end]
  #   x_B_C_T_H_W = rearrange(
  #       x_gen,
  #       "b (f h w) (pt ph pw c) -> b c (f pt) (h ph) (w pw)",
  #       f=f_gen, h=h_gen, w=w_gen, pt=patch_size[0], ph=patch_size[1],
  #       pw=patch_size[2], c=out_dim,
  #   )
"""

from typing import Tuple

import torch
import torch.amp as amp
import torch.nn.functional as F
from einops import rearrange

from lightx2v.models.networks.lyra2.infer.module_io import Lyra2PreInferOutput


class Lyra2PostInfer:
    """
    Runs the Head (modulate + norm + project) and unpatches the output tokens
    back to video latent shape.

    Source:
      Head.forward          wan2pt1.py L556-565
      Lyra2WanModel.forward wan2pt1_lyra2.py L1003-1017
    """

    def __init__(self, config: dict):
        self.config = config
        self.patch_size: Tuple[int, int, int] = tuple(config.get("patch_size", [1, 2, 2]))
        self.out_dim: int = config.get("out_dim", 16)
        self.eps: float = config.get("eps", 1e-6)

    def infer(
        self,
        weights,  # Lyra2PostWeights
        x_B_L_D: torch.Tensor,
        pre_infer_out: Lyra2PreInferOutput,
    ) -> torch.Tensor:
        """
        Args:
            weights        : Lyra2PostWeights
            x_B_L_D        : [B, L, dim]  final hidden states from transformer
            pre_infer_out  : carries e_B_D, gen_range, gen_grid

        Returns:
            x_B_C_T_H_W  : [B, out_dim, T', H', W']  predicted velocity
        """
        e_B_D = pre_infer_out.e_B_D  # [B, dim]  (float32 from pre_infer)
        gen_start, gen_end = pre_infer_out.gen_range
        f_gen, h_gen, w_gen = pre_infer_out.gen_grid
        pt, ph, pw = self.patch_size
        B, L, D = x_B_L_D.shape

        # ------ Head.forward ------
        # Original (wan2pt1.py L556-565):
        #   assert e.dtype == torch.float32
        #   e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        #   x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        with amp.autocast("cuda", dtype=torch.float32):
            modulation = weights.head_modulation.tensor  # [1, 2, dim]
            e = (modulation + e_B_D.unsqueeze(1)).chunk(2, dim=1)  # [B, 1, dim] x2
            # norm (WanLayerNorm, elementwise_affine=False)
            x_norm = F.layer_norm(x_B_L_D.float(), (D,), weight=None, bias=None, eps=self.eps)
            x_modulated = (x_norm * (1.0 + e[1]) + e[0]).to(x_B_L_D.dtype)

        # head.head linear: [B, L, dim] → [B, L, pt*ph*pw*out_dim]
        x_proj = weights.head_linear.apply(x_modulated.reshape(B * L, D)).reshape(B, L, pt * ph * pw * self.out_dim)

        # ------ Select generation tokens and unpatch ------
        # Original (wan2pt1_lyra2.py L1006-1017):
        #   x_gen = x_B_L_D[:, gen_start:gen_end]
        #   x = rearrange(x_gen, "b (f h w) (pt ph pw c) -> b c (f pt) (h ph) (w pw)", ...)
        x_gen = x_proj[:, gen_start:gen_end]  # [B, f*h*w, pt*ph*pw*out_dim]
        x_B_C_T_H_W = rearrange(
            x_gen,
            "b (f h w) (pt ph pw c) -> b c (f pt) (h ph) (w pw)",
            f=f_gen,
            h=h_gen,
            w=w_gen,
            pt=pt,
            ph=ph,
            pw=pw,
            c=self.out_dim,
        )
        return x_B_C_T_H_W
