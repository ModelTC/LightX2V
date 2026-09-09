import torch

from lightx2v.common.ops.rope import RopeTemplate
from lightx2v.models.networks.minimax_h3.infer.sglang_fused import (
    _apply_qk_neox_rope_local,
    _prepare_qk_neox_rope_inputs,
)
from lightx2v.utils.registry_factory import ROPE_REGISTER


@ROPE_REGISTER("h3ref_sgl_rope")
class MiniMaxH3SGLRope(RopeTemplate):
    def __init__(self, layout="split_half", compute_dtype=torch.bfloat16):
        if layout != "split_half":
            raise ValueError("MiniMax-H3 reference RoPE requires split_half layout")
        super().__init__(layout=layout, compute_dtype=compute_dtype)

    def prepare_freqs(self, freqs, rotary_dim: int | None = None):
        if not isinstance(freqs, tuple) or len(freqs) != 2:
            raise TypeError("MiniMax-H3 reference RoPE expects a (cos, sin) tuple")
        cos, sin = freqs
        if cos.shape != sin.shape or cos.device != sin.device:
            raise ValueError(f"MiniMax-H3 RoPE cos/sin tensors must match, got {cos.shape} and {sin.shape}")
        if cos.ndim == 2:
            if cos.shape[-1] % 2:
                raise ValueError(f"MiniMax-H3 RoPE width must be even, got {cos.shape[-1]}")
            half = cos.shape[-1] // 2
            cache = torch.cat((cos[:, :half], sin[:, :half]), dim=-1)
        elif cos.ndim == 4 and cos.shape[0] == 1 and cos.shape[2] == 1:
            if cos.shape[-1] % 2:
                raise ValueError(f"MiniMax-H3 VAE RoPE width must be even, got {cos.shape[-1]}")
            half = cos.shape[-1] // 2
            cache = torch.cat((cos[0, :, 0, :half], sin[0, :, 0, :half]), dim=-1)
        else:
            raise ValueError(f"Unsupported MiniMax-H3 reference RoPE frequency shape {cos.shape}")
        cache = cache.to(dtype=self.compute_dtype).contiguous()
        positions = torch.arange(cache.shape[0], device=cache.device, dtype=torch.long)
        return cache, positions

    @staticmethod
    def _is_prepared(freqs) -> bool:
        return isinstance(freqs, tuple) and len(freqs) == 2 and torch.is_tensor(freqs[0]) and torch.is_tensor(freqs[1]) and freqs[0].ndim == 2 and freqs[1].dtype == torch.long

    def apply(self, q: torch.Tensor, k: torch.Tensor, freqs, **kwargs):
        if kwargs.get("materialize", False):
            q, k = q.contiguous(), k.contiguous()
        if not self._is_prepared(freqs):
            freqs = self.prepare_freqs(freqs, rotary_dim=kwargs.get("rotary_dim"))
        cache, positions = freqs
        return _apply_qk_neox_rope_local(q, k, cache, positions)

    def validate_inputs(self, q: torch.Tensor, k: torch.Tensor, freqs):
        if not self._is_prepared(freqs):
            freqs = self.prepare_freqs(freqs)
        cache, positions = freqs
        return _prepare_qk_neox_rope_inputs(q, k, cache, positions)

    def apply_single(self, x: torch.Tensor, freqs, **kwargs) -> torch.Tensor:
        return self.apply(x, torch.empty_like(x), freqs, **kwargs)[0]


__all__ = ["MiniMaxH3SGLRope"]
