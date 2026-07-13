from __future__ import annotations

import torch

from lightx2v.common.magi_custom_op_mode import use_magi_custom_ops
from lightx2v.utils.registry_factory import ROPE_REGISTER

try:
    from magi_compiler import magi_register_custom_op
except ImportError:
    magi_register_custom_op = None

from .template import RopeTemplate

try:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
except ImportError:
    apply_rope_with_cos_sin_cache_inplace = None


@torch.library.custom_op(
    "lightx2v::rope_flashinfer_",
    mutates_args=("query", "key"),
    device_types="cuda",
    schema=("(Tensor positions, Tensor(a!) query, Tensor(b!) key, Tensor cos_sin_cache, int head_size, bool is_neox) -> ()"),
)
def rope_flashinfer_(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    is_neox: bool,
) -> None:
    apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )


@rope_flashinfer_.register_fake
def rope_flashinfer_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    is_neox: bool,
) -> None:
    return None


@ROPE_REGISTER("flashinfer_rope")
class FlashInferRope(RopeTemplate):
    def _apply_eager(self, q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor, positions: torch.Tensor | None = None):
        if apply_rope_with_cos_sin_cache_inplace is None:
            raise ImportError("flashinfer is required for FlashInferRope.")
        if q.ndim != 3 or k.ndim != 3:
            raise ValueError(f"FlashInferRope expects [L, H, D] tensors, got q={q.shape}, k={k.shape}.")
        length, q_heads, head_dim = q.shape
        k_heads = k.shape[1]
        if positions is None:
            positions = torch.arange(length, device=q.device, dtype=torch.long)
        query = q.reshape(length, q_heads * head_dim).contiguous()
        key = k.reshape(length, k_heads * head_dim).contiguous()
        is_neox = self.layout == "split_half"
        if torch.compiler.is_compiling():
            rope_flashinfer_(positions, query, key, freqs, head_dim, is_neox)
        else:
            apply_rope_with_cos_sin_cache_inplace(
                positions=positions,
                query=query,
                key=key,
                head_size=head_dim,
                cos_sin_cache=freqs,
                is_neox=is_neox,
            )
        return query.view_as(q), key.view_as(k)

    def apply(self, q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor, positions: torch.Tensor | None = None, **kwargs):
        if positions is None and self.layout == "interleaved" and use_magi_custom_ops() and magi_register_custom_op is not None and apply_rope_with_cos_sin_cache_inplace is not None:
            return torch.ops.lightx2v.rope_flashinfer(q, k, freqs)
        return self._apply_eager(q, k, freqs, positions=positions)

    def apply_single(self, x: torch.Tensor, freqs, **kwargs):
        output, _ = self.apply(x, x.clone(), freqs, **kwargs)
        return output


def _rope_meta(q, k, freqs):
    return torch.empty_like(q), torch.empty_like(k)


if magi_register_custom_op is not None and apply_rope_with_cos_sin_cache_inplace is not None:

    @magi_register_custom_op(
        "lightx2v::rope_flashinfer",
        infer_output_meta_fn=_rope_meta,
        is_subgraph_boundary=True,
    )
    def _rope_flashinfer_custom_op(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor):
        return FlashInferRope(layout="interleaved")._apply_eager(q.clone(), k.clone(), freqs)
