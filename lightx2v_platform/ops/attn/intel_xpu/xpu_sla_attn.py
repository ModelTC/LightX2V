"""Intel XPU adapters for the SLA routing and sparse-attention kernels."""

from __future__ import annotations

try:
    import sycl_kernels as _sycl_kernels
except (ImportError, OSError) as exc:
    _sycl_kernels = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _get_sycl_api(name):
    if _sycl_kernels is None:
        raise RuntimeError(
            "Intel XPU SLA requires lightx2v_kernel_xpu's sycl-kernels package "
            "with SLA/CUTE sparse attention enabled"
        ) from _IMPORT_ERROR
    api = getattr(_sycl_kernels, name, None)
    if api is None:
        raise RuntimeError(
            f"sycl_kernels.{name} is unavailable; rebuild/install "
            "lightx2v_kernel_xpu with SLA/CUTE sparse attention enabled"
        )
    return api


def sla_block_map(q, k, keep_ratio=0.2, block_q=128, block_k=128):
    """Build an SLA LUT for BLHD tensors using ``sycl_kernels.sla_block_map``."""
    return _get_sycl_api("sla_block_map")(q, k, keep_ratio, block_q, block_k)


def sparse_block_attention(q, k, v, lut, block_q=128, block_k=128, scale=None):
    """Apply fused sparse QK/softmax/PV using a precomputed SLA LUT."""
    return _get_sycl_api("sparse_block_attention")(q, k, v, lut, block_q, block_k, scale)


def sla_sparse_attention(q, k, v, keep_ratio=0.2, block_q=128, block_k=128, scale=None):
    """Route and apply sparse attention through ``sycl_kernels.sla_sparse_attention``."""
    return _get_sycl_api("sla_sparse_attention")(q, k, v, keep_ratio, block_q, block_k, scale)
