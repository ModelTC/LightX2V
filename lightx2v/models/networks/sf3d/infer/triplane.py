# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from __future__ import annotations

from lightx2v.models.networks.sf3d.weights.triplane import Float, Tensor, rearrange, repeat


def TriplaneLearnablePositionalEmbedding_forward(self, batch_size: int) -> Float[Tensor, "B Ct Nt"]:
    return rearrange(repeat(self.embeddings, "Np Ct Hp Wp -> B Np Ct Hp Wp", B=batch_size), "B Np Ct Hp Wp -> B Ct (Np Hp Wp)")


def TriplaneLearnablePositionalEmbedding_detokenize(self, tokens: Float[Tensor, "B Ct Nt"]) -> Float[Tensor, "B 3 Ct Hp Wp"]:
    _batch_size, Ct, Nt = tokens.shape
    assert Nt == self.cfg.plane_size**2 * 3
    assert Ct == self.cfg.num_channels
    return rearrange(tokens, "B Ct (Np Hp Wp) -> B Np Ct Hp Wp", Np=3, Hp=self.cfg.plane_size, Wp=self.cfg.plane_size)
