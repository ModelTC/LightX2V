import torch

from lightx2v_platform.registry_factory import PLATFORM_QKV_NORM_REGISTER


def _can_use_xpu_kernel(packed, norm_q, norm_k, head_dim):
    if not (
        packed.device.type == "xpu"
        and head_dim == 128
        and packed.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and all(
            getattr(norm, "weight", None) is not None
            and norm.weight.device == packed.device
            and norm.weight.dtype == packed.dtype
            and norm.weight.is_contiguous()
            and norm.sensitive_layer_dtype == norm.infer_dtype
            for norm in (norm_q, norm_k)
        )
    ):
        return False
    try:
        import sycl_kernels

        return sycl_kernels.has_minimax_h3_qkv_norm()
    except (AttributeError, ImportError, OSError, RuntimeError):
        return False


@PLATFORM_QKV_NORM_REGISTER("int_xpu")
class XpuQKVNorm:
    @staticmethod
    def apply(packed, norm_q, norm_k, num_heads, head_dim):
        if not _can_use_xpu_kernel(packed, norm_q, norm_k, head_dim):
            # Import lazily to keep the platform layer independent from model
            # modules during registry initialization.
            from lightx2v.models.networks.minimax_h3.infer.fused_qkv import TritonQKVNorm

            return TritonQKVNorm.apply(packed, norm_q, norm_k, num_heads, head_dim)

        import sycl_kernels

        return sycl_kernels.minimax_h3_qkv_norm(
            packed, norm_q.weight, norm_k.weight, norm_q.eps, norm_k.eps
        )
