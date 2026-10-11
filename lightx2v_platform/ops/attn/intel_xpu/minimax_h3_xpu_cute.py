import torch
from loguru import logger

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER


@torch.library.custom_op("lightx2v::minimax_h3_vae_sdp_d64", mutates_args=())
def _minimax_h3_vae_sdp_d64(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    import sycl_kernels

    return sycl_kernels.minimax_h3_vae_sdp_d64(q, k, v)


@_minimax_h3_vae_sdp_d64.register_fake
def _minimax_h3_vae_sdp_d64_fake(q, k, v):
    batch, heads, sequence, dim = q.shape
    return q.new_empty((batch, sequence, heads, dim)).permute(0, 2, 1, 3)


def _has_native_kernel():
    try:
        import sycl_kernels

        return sycl_kernels.has_minimax_h3_vae_sdp_d64()
    except (ImportError, OSError, RuntimeError):
        return False


def _has_supported_layout(q, k, v):
    if q.ndim != 4 or tuple(k.shape) != tuple(q.shape) or tuple(v.shape) != tuple(q.shape):
        return False
    batch, heads, sequence, dim = q.shape
    if batch != 1 or heads != 32 or dim != 64 or sequence <= 5:
        return False
    qk_stride = (sequence * heads * dim, dim, heads * dim, 1)
    packed_v_stride = (3 * sequence * heads * dim, 3 * dim, 3 * heads * dim, 1)
    return tuple(q.stride()) == qk_stride and tuple(k.stride()) == qk_stride and tuple(v.stride()) in (qk_stride, packed_v_stride)


@PLATFORM_ATTN_WEIGHT_REGISTER("minimax_h3_xpu_cute")
class MiniMaxH3XpuCuteWeight(AttnWeightTemplate):
    """BMG CUTE D64 attention with a Torch SDPA fallback."""

    _backend_status_logged = False

    def __init__(self):
        # Import lazily so platform ops finish registering before the framework
        # registry takes its one-time snapshot of platform implementations.
        from lightx2v.common.ops.attn.torch_sdpa import TorchSDPAWeight

        self.config = {}
        self.native_available = _has_native_kernel()
        self.fallback = TorchSDPAWeight()
        if not type(self)._backend_status_logged:
            type(self)._backend_status_logged = True
            if self.native_available:
                logger.info("MiniMax-H3 Video VAE CUTE D64 attention enabled")
            else:
                logger.warning("MiniMax-H3 Video VAE CUTE D64 kernel unavailable; using Torch SDPA")

    def apply(
        self,
        q,
        k,
        v,
        drop_rate=0,
        attn_mask=None,
        causal=False,
        **kwargs,
    ):
        q_bhld = q.unsqueeze(0) if q.ndim == 3 else q
        k_bhld = k.unsqueeze(0) if k.ndim == 3 else k
        v_bhld = v.unsqueeze(0) if v.ndim == 3 else v
        q_bhld = q_bhld.transpose(1, 2)
        k_bhld = k_bhld.transpose(1, 2)
        v_bhld = v_bhld.transpose(1, 2)

        if (
            self.native_available
            and drop_rate == 0
            and attn_mask is None
            and not causal
            and q_bhld.device.type == "xpu"
            and q_bhld.dtype == torch.float16
            and k_bhld.dtype == q_bhld.dtype
            and v_bhld.dtype == q_bhld.dtype
            and _has_supported_layout(q_bhld, k_bhld, v_bhld)
        ):
            output = _minimax_h3_vae_sdp_d64(q_bhld, k_bhld, v_bhld)
            output = output.transpose(1, 2).reshape(q_bhld.shape[0], q_bhld.shape[2], -1)
            return output.squeeze(0)

        return self.fallback.apply(
            q,
            k,
            v,
            drop_rate=drop_rate,
            attn_mask=attn_mask,
            causal=causal,
            **kwargs,
        )
