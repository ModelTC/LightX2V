import torch
from loguru import logger

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

# Try to import Flash Attention (ROCm version 2.6.1)
try:
    from flash_attn import flash_attn_varlen_func

    FLASH_ATTN_AVAILABLE = True
    logger.info(f"Flash Attention (ROCm) is available")
except ImportError:
    logger.warning("Flash Attention not found. Will use PyTorch SDPA as fallback.")
    flash_attn_varlen_func = None
    FLASH_ATTN_AVAILABLE = False


@PLATFORM_ATTN_WEIGHT_REGISTER("flash_attn_hygon_dcu")
class FlashAttnHygonDcu(AttnWeightTemplate):
    """
    Hygon DCU Flash Attention implementation.

    Uses AMD ROCm version of Flash Attention 2.6.1 when available.
    Falls back to PyTorch SDPA (Scaled Dot Product Attention) if Flash Attention is not installed.

    Tested Environment:
    - PyTorch: 2.7.1
    - Python: 3.10
    - Flash Attention: 2.6.1 (ROCm)
    Reference: https://developer.sourcefind.cn/codes/modelzoo/wan2.1_pytorch/-/blob/master/wan/modules/attention.py
    """

    def __init__(self, weight_name="flash_attn_hygon_dcu"):
        super().__init__(weight_name)
        self.use_flash_attn = FLASH_ATTN_AVAILABLE

        if self.use_flash_attn:
            logger.info("Flash Attention 2.6.1 (ROCm) is available and will be used.")
        else:
            logger.warning("Flash Attention not available. Using PyTorch SDPA fallback.")

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        model_cls=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
    ):
        """
        Execute Flash Attention computation with variable-length sequences.
        
        This method signature matches the standard LightX2V attention interface,
        compatible with other platform implementations (e.g., MLU, NVIDIA).
        
        Args:
            q: [B*Lq, Nq, C1] Query tensor (flattened batch)
            k: [B*Lk, Nk, C1] Key tensor (flattened batch)
            v: [B*Lk, Nk, C2] Value tensor (flattened batch)
            cu_seqlens_q: [B+1] Cumulative sequence lengths for queries
            cu_seqlens_kv: [B+1] Cumulative sequence lengths for keys/values
            max_seqlen_q: Maximum sequence length in queries
            max_seqlen_kv: Maximum sequence length in keys/values
            model_cls: Model class identifier (unused but kept for interface compatibility)
            dropout_p: Dropout probability
            softmax_scale: Scaling factor for QK^T before softmax
            causal: Whether to apply causal mask
            window_size: Sliding window size tuple (left, right)
            deterministic: Whether to use deterministic algorithm
        Returns:
            Output tensor: [B*Lq, C2] (flattened batch)
        """
        if not self.use_flash_attn:
            # Fallback to PyTorch SDPA
            return self._sdpa_fallback(q, k, v, cu_seqlens_q, max_seqlen_q, causal, dropout_p)

        # Ensure data types are half precision
        import math
        half_dtypes = (torch.float16, torch.bfloat16)
        dtype = q.dtype if q.dtype in half_dtypes else torch.bfloat16
        out_dtype = q.dtype

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        # Convert to half precision
        q_flat = half(q)
        k_flat = half(k)
        v_flat = half(v)

        # Compute softmax scale if not provided
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        # Use Flash Attention 2.6.1 (ROCm version) with varlen interface
        output = flash_attn_varlen_func(
            q=q_flat,
            k=k_flat,
            v=v_flat,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_kv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        )

        # Reshape to [B*max_seqlen_q, num_heads * head_dim]
        bs = cu_seqlens_q.shape[0] - 1
        output = output.reshape(bs * max_seqlen_q, -1)
        return output.to(out_dtype)

    def _sdpa_fallback(self, q, k, v, cu_seqlens_q, max_seqlen_q, causal=False, dropout_p=0.0):
        """
        Fallback to PyTorch Scaled Dot Product Attention when Flash Attention is not available.
        
        Args:
            q: [B*Lq, Nq, C] Query tensor (flattened batch)
            k: [B*Lk, Nk, C] Key tensor (flattened batch)
            v: [B*Lk, Nk, C] Value tensor (flattened batch)
            cu_seqlens_q: [B+1] Cumulative sequence lengths for queries
            max_seqlen_q: Maximum sequence length in queries
            causal: Whether to apply causal mask
            dropout_p: Dropout probability
        Returns:
            Output tensor: [B*Lq, C] (flattened batch)
        """
        # Reshape from flattened format to batched format
        bs = cu_seqlens_q.shape[0] - 1
        
        # Reshape q, k, v to [B, L, Nq, C]
        q = q.reshape(bs, max_seqlen_q, q.shape[-2], q.shape[-1])
        k = k.reshape(bs, max_seqlen_q, k.shape[-2], k.shape[-1])
        v = v.reshape(bs, max_seqlen_q, v.shape[-2], v.shape[-1])
        
        # Transpose to [B, Nq, L, C] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p
        )

        # Transpose back to [B, L, Nq, C] and flatten
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(bs * max_seqlen_q, -1)
        
        return out
