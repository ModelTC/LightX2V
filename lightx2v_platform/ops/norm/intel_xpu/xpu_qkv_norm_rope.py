from lightx2v_platform.registry_factory import PLATFORM_QKV_NORM_ROPE_REGISTER

@PLATFORM_QKV_NORM_ROPE_REGISTER("intel_xpu")
class XpuQKVNormRope:
    @staticmethod
    def apply(packed, norm_q, norm_k, rope, freqs):
        # Import lazily to avoid a platform/model import cycle during registry setup.
        from lightx2v.models.networks.minimax_h3.infer.fused_qkv import (
            prepare_qkv_norm_rope,
            run_triton_qkv_norm_rope,
        )

        prepared = prepare_qkv_norm_rope(packed, norm_q, norm_k, rope, freqs)
        if prepared is None:
            return None
        cos, sin, low_precision = prepared
        if packed.device.type == "xpu" and norm_q.weight.numel() == 128 and cos.shape[1] == 96:
            try:
                import sycl_kernels
            except ImportError:
                sycl_kernels = None
            if sycl_kernels is not None and getattr(sycl_kernels, "has_minimax_h3_qkv_norm_rope", lambda: False)():
                return sycl_kernels.minimax_h3_qkv_norm_rope(
                    packed,
                    norm_q.weight,
                    norm_k.weight,
                    cos,
                    sin,
                    norm_q.eps,
                    norm_k.eps,
                    low_precision,
                )
        return run_triton_qkv_norm_rope(packed, norm_q, norm_k, cos, sin, low_precision)
