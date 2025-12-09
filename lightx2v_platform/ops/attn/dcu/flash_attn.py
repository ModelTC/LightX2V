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


@PLATFORM_ATTN_WEIGHT_REGISTER("flash_attn_dcu")
class FlashAttnDcu(AttnWeightTemplate):
    """
    DCU Flash Attention implementation.
    
    Uses AMD ROCm version of Flash Attention 2.6.1 when available.
    Falls back to PyTorch SDPA (Scaled Dot Product Attention) if Flash Attention is not installed.
    
    Tested Environment:
    - PyTorch: 2.7.1
    - Python: 3.10
    - Flash Attention: 2.6.1 (ROCm)
    
    Reference: https://developer.sourcefind.cn/codes/modelzoo/wan2.1_pytorch/-/blob/master/wan/modules/attention.py
    """

    def __init__(self, weight_name="flash_attn_dcu"):
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
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
    ):
        """
        Execute Flash Attention computation.
        
        Args:
            q: [B, Lq, Nq, C1] Query tensor
            k: [B, Lk, Nk, C1] Key tensor
            v: [B, Lk, Nk, C2] Value tensor
            q_lens: [B] Optional sequence lengths for queries
            k_lens: [B] Optional sequence lengths for keys
            dropout_p: Dropout probability
            softmax_scale: Scaling factor for QK^T before softmax
            causal: Whether to apply causal mask
            window_size: Sliding window size tuple (left, right)
            deterministic: Whether to use deterministic algorithm
            
        Returns:
            Output tensor: [B, Lq, Nq, C2]
        """
        if not self.use_flash_attn:
            # Fallback to PyTorch SDPA
            return self._sdpa_fallback(q, k, v, causal, dropout_p)

        # Ensure data types are half precision
        half_dtypes = (torch.float16, torch.bfloat16)
        dtype = q.dtype if q.dtype in half_dtypes else torch.bfloat16
        out_dtype = q.dtype

        b, lq, lk = q.size(0), q.size(1), k.size(1)

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        # Preprocess query
        if q_lens is None:
            q_flat = half(q.flatten(0, 1))
            q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
        else:
            q_flat = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

        # Preprocess key/value
        if k_lens is None:
            k_flat = half(k.flatten(0, 1))
            v_flat = half(v.flatten(0, 1))
            k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
        else:
            k_flat = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
            v_flat = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

        # Compute cumulative sequence lengths
        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)

        # Use Flash Attention 2.6.1 (ROCm version)
        output = flash_attn_varlen_func(
            q=q_flat,
            k=k_flat,
            v=v_flat,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        )

        # Reshape back to batch dimension
        output = output.unflatten(0, (b, lq))
        return output.to(out_dtype)

    def _sdpa_fallback(self, q, k, v, causal=False, dropout_p=0.0):
        """
        Fallback to PyTorch Scaled Dot Product Attention.
        
        Args:
            q: [B, Lq, Nq, C] Query tensor
            k: [B, Lk, Nk, C] Key tensor
            v: [B, Lk, Nk, C] Value tensor
            causal: Whether to apply causal mask
            dropout_p: Dropout probability
            
        Returns:
            Output tensor: [B, Lq, Nq, C]
        """
        # Transpose to [B, Nq, Lq, C] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p
        )

        # Transpose back to [B, Lq, Nq, C]
        return out.transpose(1, 2).contiguous()

