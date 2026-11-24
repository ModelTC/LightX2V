# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from __future__ import annotations

from lightx2v.models.networks.hymotion.weights.positional_encoding import Tensor, Tuple, torch


def RotaryEmbedding_apply_rotary_emb(self, xq: Tensor, xk: Tensor) -> Tuple[Tensor, Tensor]:
    from lightx2v.utils.registry_factory import ROPE_REGISTER

    if not hasattr(self, "_registered_rope"):
        self._registered_rope = ROPE_REGISTER["torch_real_rope" if isinstance(self.freqs_cis, tuple) else "torch_complex_rope"](layout="interleaved", compute_dtype=torch.float32)
    length = xq.shape[1]
    if isinstance(self.freqs_cis, tuple):
        freqs = tuple(t[:length].to(xq.device)[None, :, None, :] for t in self.freqs_cis)
    else:
        freqs = self.freqs_cis[:length].to(xq.device)[None, :, None, :]
    return self._registered_rope.apply(xq, xk, freqs)
