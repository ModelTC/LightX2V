from __future__ import annotations

import torch

from lightx2v.common.magi_custom_op_mode import use_magi_custom_ops
from lightx2v.utils.registry_factory import ROPE_REGISTER

try:
    from magi_compiler import magi_register_custom_op
except ImportError:
    magi_register_custom_op = None

from .template import RopeTemplate, broadcast_freqs


@ROPE_REGISTER("torch_complex_rope")
class TorchComplexRope(RopeTemplate):
    def __init__(self, compute_dtype: torch.dtype = torch.float32):
        super().__init__(layout="interleaved", compute_dtype=compute_dtype)

    def apply(self, q: torch.Tensor, k: torch.Tensor, freqs, **kwargs):
        if q.ndim == 3 and k.ndim == 3 and self.compute_dtype == torch.float32 and torch.is_complex(freqs) and use_magi_custom_ops() and magi_register_custom_op is not None and not kwargs:
            return torch.ops.lightx2v.rope_torch_complex(q, k, freqs)
        return super().apply(q, k, freqs, **kwargs)

    def apply_single(self, x: torch.Tensor, freqs: torch.Tensor, rotary_dim: int | None = None, unsqueeze_dim: int = -2):
        if not torch.is_complex(freqs):
            raise TypeError("TorchComplexRope expects a complex frequency tensor.")
        rotary_dim = rotary_dim or x.shape[-1]
        if rotary_dim % 2:
            raise ValueError(f"rotary_dim must be even, got {rotary_dim}.")
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        x_complex = torch.view_as_complex(x_rot.to(self.compute_dtype).reshape(*x_rot.shape[:-1], -1, 2).contiguous())
        freqs = broadcast_freqs(freqs, x_complex, unsqueeze_dim)
        output = torch.view_as_real(x_complex * freqs).flatten(-2).to(x.dtype)
        return torch.cat((output, x_pass), dim=-1) if x_pass.shape[-1] else output


@ROPE_REGISTER("torch_real_rope")
class TorchRealRope(RopeTemplate):
    def _cos_sin(self, freqs, rotary_dim: int):
        if torch.is_tensor(freqs) and torch.is_complex(freqs):
            return freqs.real, freqs.imag, True
        if isinstance(freqs, tuple):
            cos, sin = freqs
            return cos, sin, cos.shape[-1] == rotary_dim // 2
        if torch.is_tensor(freqs):
            if freqs.shape[-1] != rotary_dim:
                raise ValueError(f"Concatenated cos-sin cache must have last dim {rotary_dim}, got {freqs.shape[-1]}.")
            return freqs[..., : rotary_dim // 2], freqs[..., rotary_dim // 2 :], True
        raise TypeError(f"Unsupported RoPE frequency type: {type(freqs)!r}")

    @staticmethod
    def _rotate_interleaved(x: torch.Tensor):
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        first, second = pairs.unbind(dim=-1)
        return torch.stack((-second, first), dim=-1).flatten(-2)

    @staticmethod
    def _rotate_split_half(x: torch.Tensor):
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def apply_single(self, x: torch.Tensor, freqs, rotary_dim: int | None = None, unsqueeze_dim: int = -2):
        rotary_dim = rotary_dim or x.shape[-1]
        if rotary_dim % 2:
            raise ValueError(f"rotary_dim must be even, got {rotary_dim}.")
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        cos, sin, pairwise = self._cos_sin(freqs, rotary_dim)
        x_float = x_rot.to(self.compute_dtype)
        if pairwise:
            if self.layout == "interleaved":
                first, second = x_float[..., 0::2], x_float[..., 1::2]
            else:
                first, second = x_float.chunk(2, dim=-1)
            cos = broadcast_freqs(cos.to(self.compute_dtype), first, unsqueeze_dim)
            sin = broadcast_freqs(sin.to(self.compute_dtype), first, unsqueeze_dim)
            first_out = first * cos - second * sin
            second_out = first * sin + second * cos
            if self.layout == "interleaved":
                output = torch.empty_like(x_float)
                output[..., 0::2] = first_out
                output[..., 1::2] = second_out
            else:
                output = torch.cat((first_out, second_out), dim=-1)
        else:
            cos = broadcast_freqs(cos.to(self.compute_dtype), x_float, unsqueeze_dim)
            sin = broadcast_freqs(sin.to(self.compute_dtype), x_float, unsqueeze_dim)
            rotate = self._rotate_interleaved if self.layout == "interleaved" else self._rotate_split_half
            output = x_float * cos + rotate(x_float) * sin
        output = output.to(x.dtype)
        return torch.cat((output, x_pass), dim=-1) if x_pass.shape[-1] else output


def _rope_meta(q, k, freqs):
    return torch.empty_like(q), torch.empty_like(k)


if magi_register_custom_op is not None:

    @magi_register_custom_op(
        "lightx2v::rope_torch_complex",
        infer_output_meta_fn=_rope_meta,
        is_subgraph_boundary=True,
    )
    def _rope_torch_complex_custom_op(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor):
        module = TorchComplexRope()
        return module.apply_single(q, freqs), module.apply_single(k, freqs)
