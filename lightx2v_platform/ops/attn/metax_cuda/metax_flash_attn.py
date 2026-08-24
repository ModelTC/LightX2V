import torch

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except (ImportError, OSError):
    flash_attn_func = None
    flash_attn_varlen_func = None


@torch.library.custom_op(
    "lightx2v::metax_flash_attn2",
    mutates_args=(),
    device_types="cuda",
)
def metax_flash_attn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None,
    causal: bool,
) -> torch.Tensor:
    return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=softmax_scale, causal=causal)


@metax_flash_attn2.register_fake
def _metax_flash_attn2_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None,
    causal: bool,
) -> torch.Tensor:
    return q.new_empty((*q.shape[:-1], v.shape[-1]))


@torch.library.custom_op(
    "lightx2v::metax_flash_attn2_varlen",
    mutates_args=(),
    device_types="cuda",
)
def metax_flash_attn2_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None,
    causal: bool,
) -> torch.Tensor:
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
    )


@metax_flash_attn2_varlen.register_fake
def _metax_flash_attn2_varlen_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None,
    causal: bool,
) -> torch.Tensor:
    return q.new_empty((*q.shape[:-1], v.shape[-1]))


@torch.library.custom_op(
    "lightx2v::metax_flash_attn2_with_lse",
    mutates_args=(),
    device_types="cuda",
)
def metax_flash_attn2_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, lse, *_ = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
        return_attn_probs=True,
    )
    return output, lse


@metax_flash_attn2_with_lse.register_fake
def _metax_flash_attn2_with_lse_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = q.new_empty((*q.shape[:-1], v.shape[-1]))
    lse = q.new_empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32)
    return output, lse


@PLATFORM_ATTN_WEIGHT_REGISTER("metax_flash_attn2")
class MetaxFlashAttn2Weight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}
        if flash_attn_func is None:
            raise RuntimeError("metax_flash_attn2 requires the MetaX flash-attn package")

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        if kwargs.get("dropout_p", 0.0) != 0.0:
            raise ValueError("metax_flash_attn2 only supports dropout=0")

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        softmax_scale = kwargs.get("softmax_scale")
        causal = kwargs.get("causal", False)

        use_varlen = q.ndim == 3 and cu_seqlens_q is not None and cu_seqlens_q.shape[0] > 2
        if use_varlen:
            cu_seqlens_q = cu_seqlens_q.to(q.device, dtype=torch.int32).contiguous()
            cu_seqlens_kv = cu_seqlens_kv.to(k.device, dtype=torch.int32).contiguous()
            output = metax_flash_attn2_varlen(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                softmax_scale,
                causal,
            )
        else:
            if q.ndim == 3:
                q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
            output = metax_flash_attn2(q, k, v, softmax_scale, causal)

        return output.reshape(-1, output.shape[-2] * output.shape[-1])

    def apply_with_lse(self, q, k, v, softmax_scale=None):
        if q.ndim == 3:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        output, lse = metax_flash_attn2_with_lse(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            softmax_scale,
        )
        output = output.reshape(q.shape[0] * q.shape[1], -1)
        lse = lse.transpose(1, 2).reshape(q.shape[0] * q.shape[1], q.shape[2])
        return output, lse


__all__ = ["MetaxFlashAttn2Weight"]
