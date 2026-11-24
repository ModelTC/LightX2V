# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from __future__ import annotations

from lightx2v.models.networks.hymotion.weights.positional_encoding import Tensor, Tuple, Union, torch


def RotaryEmbedding_reshape_for_broadcast(self, freqs_cis: Union[Tensor, Tuple[Tensor, Tensor]], x: Tensor) -> Union[Tuple[Tensor, Tensor], Tensor]:
    ndim = x.ndim
    assert ndim > 1
    if isinstance(freqs_cis, tuple):
        assert freqs_cis[0].shape[-1] == x.shape[-1], f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape} on the head_dim dimension"
        assert freqs_cis[0].shape[0] >= x.shape[1], f"freqs_cis shape {freqs_cis[0].shape} should be larger than or equal to x shape {x.shape} on the time dimension"
        shape = []
        for i, d in enumerate(x.shape):
            if i == 1:
                shape.append(-1)
            elif i == ndim - 1:
                shape.append(d)
            else:
                shape.append(1)
        return (freqs_cis[0].view(*shape)[:, : x.shape[1], ...], freqs_cis[1].view(*shape)[:, : x.shape[1], ...])
    else:
        assert freqs_cis.shape[-1] == x.shape[-1], f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape} on the head_dim dimension"
        assert freqs_cis.shape[0] >= x.shape[1], f"freqs_cis shape {freqs_cis.shape} should be larger than or equal to x shape {x.shape} on the time dimension"
        shape = []
        for i, d in enumerate(x.shape):
            if i == 1:
                shape.append(-1)
            elif i == ndim - 1:
                shape.append(d)
            else:
                shape.append(1)
        return freqs_cis.view(*shape)[:, : x.shape[1], ...]


def RotaryEmbedding_rotate_half(self, x: Tensor) -> Tensor:
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


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


def PositionalEncoding_forward(self, x: Tensor) -> Tensor:
    x = x + self.pe[:, : x.shape[1], :]
    return self.dropout(x)
