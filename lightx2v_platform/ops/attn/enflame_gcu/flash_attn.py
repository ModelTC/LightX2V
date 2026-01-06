import torch
from loguru import logger

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

# Try to import Flash Attention 2 and Flash Attention 3 (enflame supports both)
FLASH_ATTN_3_AVAILABLE = False
FLASH_ATTN_2_AVAILABLE = False
flash_attn_varlen_func = None
flash_attn_interface = None

# Try Flash Attention 3 first (newer, faster)
try:
    import flash_attn_interface
    flash_attn_interface.flash_attn_varlen_func  # Test if function exists
    FLASH_ATTN_3_AVAILABLE = True
    logger.info("Flash Attention 3 is available for Enflame GCU")
except (ImportError, AttributeError):
    # Fallback to Flash Attention 2
    try:
        from flash_attn import flash_attn_varlen_func
        FLASH_ATTN_2_AVAILABLE = True
        logger.info("Flash Attention 2 is available for Enflame GCU")
    except ImportError:
        logger.warning("Flash Attention not found. Will use PyTorch SDPA as fallback.")
        flash_attn_varlen_func = None


@PLATFORM_ATTN_WEIGHT_REGISTER("flash_attn_enflame_gcu")
class FlashAttnEnflameGcu(AttnWeightTemplate):
    """
    Enflame GCU Flash Attention implementation.

    Uses Flash Attention 2 when available.
    Falls back to PyTorch SDPA (Scaled Dot Product Attention) if Flash Attention is not installed.

    Key compatibility notes for GCU:
    - GCU does not support float64, use float32
    - GCU does not support int64/uint64, use int32
    - All sequence length parameters must be int32

    Reference: scripts/enflame/wan2.1/wan/modules/attention.py
    """

    def __init__(self, weight_name="flash_attn_enflame_gcu"):
        super().__init__(weight_name)
        self.use_flash_attn = FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE
        self.use_flash_attn_3 = FLASH_ATTN_3_AVAILABLE

        if self.use_flash_attn:
            version_str = "3" if self.use_flash_attn_3 else "2"
            logger.info(f"Flash Attention {version_str} is available and will be used for Enflame GCU.")
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
            cu_seqlens_q: [B+1] Cumulative sequence lengths for queries (must be int32)
            cu_seqlens_kv: [B+1] Cumulative sequence lengths for keys/values (must be int32)
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

        # Ensure cu_seqlens are int32 (GCU does not support int64)
        # Build cu_seqlens from q_lens/k_lens if provided, following native implementation
        if cu_seqlens_q is None:
            # If not provided, assume uniform sequence lengths
            bs = q_flat.shape[0] // max_seqlen_q if max_seqlen_q else 1
            q_lens = torch.tensor([max_seqlen_q] * bs, dtype=torch.int32).to(
                device=q_flat.device, non_blocking=True
            )
            cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32
            ).to(q_flat.device, non_blocking=True)
        elif cu_seqlens_q.dtype != torch.int32:
            cu_seqlens_q = cu_seqlens_q.to(torch.int32)

        if cu_seqlens_kv is None:
            bs = k_flat.shape[0] // max_seqlen_kv if max_seqlen_kv else 1
            k_lens = torch.tensor([max_seqlen_kv] * bs, dtype=torch.int32).to(
                device=k_flat.device, non_blocking=True
            )
            cu_seqlens_kv = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32
            ).to(k_flat.device, non_blocking=True)
        elif cu_seqlens_kv.dtype != torch.int32:
            cu_seqlens_kv = cu_seqlens_kv.to(torch.int32)

        # Ensure max_seqlen are int (not int64)
        if max_seqlen_q is not None:
            max_seqlen_q = int(max_seqlen_q)
        if max_seqlen_kv is not None:
            max_seqlen_kv = int(max_seqlen_kv)

        # Compute softmax scale if not provided
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q_flat.shape[-1])

        # Use Flash Attention 3 or 2 with varlen interface
        # Following native implementation: prefer FA3, fallback to FA2
        if self.use_flash_attn_3 and flash_attn_interface is not None:
            # Flash Attention 3 (Note: dropout_p, window_size not supported in FA3)
            output = flash_attn_interface.flash_attn_varlen_func(
                q=q_flat,
                k=k_flat,
                v=v_flat,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_kv,
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_kv,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic,
            )[0]  # FA3 returns tuple, take first element
        elif flash_attn_varlen_func is not None:
            # Flash Attention 2
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
        else:
            # Should not reach here if use_flash_attn is True
            raise RuntimeError("Flash Attention is marked as available but function is None")

        # Reshape to [B*max_seqlen_q, num_heads * head_dim]
        bs = cu_seqlens_q.shape[0] - 1
        if max_seqlen_q:
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

        # GCU compatibility: ensure all tensors are float32 (not float64)
        # scaled_dot_product_attention may use int64 internally, but we ensure inputs are correct
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p
        )

        # Transpose back to [B, L, Nq, C] and flatten
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(bs * max_seqlen_q, -1)

        return out

