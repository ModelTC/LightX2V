# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import warnings

import torch

FLASH_ATTN_3_IMPORT_ERROR = None
try:
    import flash_attn_interface

    # Availability describes the extension, not whichever GPU happens to be
    # current during import. Eligibility is checked on the actual query device.
    FLASH_ATTN_3_AVAILABLE = True
except (ImportError, OSError) as error:
    flash_attn_interface = None
    FLASH_ATTN_3_AVAILABLE = False
    FLASH_ATTN_3_IMPORT_ERROR = str(error)

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

__all__ = [
    "flash_attention",
    "attention",
    "sdpa_attention",
    "require_flash_attention_3",
]


def is_hopper_gpu(device=None):
    """Identify Hopper by compute capability, including both H100 and H200."""
    if device is not None and torch.device(device).type != "cuda":
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] == 9


def require_flash_attention_3(device, *, dropout_p=0.0, window_size=(-1, -1)):
    """Fail closed for explicit FA3 requests; do not silently execute FA2."""
    if dropout_p != 0.0 or tuple(window_size) != (-1, -1):
        raise ValueError("flash_attention_3 currently requires dropout_p=0 and global window_size=(-1, -1)")
    if not FLASH_ATTN_3_AVAILABLE:
        raise RuntimeError(f"flash_attention_3 was explicitly requested, but flash_attn_interface is unavailable: {FLASH_ATTN_3_IMPORT_ERROR}")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("flash_attention_3 was explicitly requested, but requires a CUDA Hopper GPU (compute capability 9.x)")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 9:
        raise RuntimeError(f"flash_attention_3 requires a Hopper GPU (compute capability 9.x), got {capability} on {device}")
    return {"device": str(device), "gpu": torch.cuda.get_device_name(device), "compute_capability": capability}


def resolve_flash_attention_version(backend, version, device, *, dropout_p=0.0, window_size=(-1, -1)):
    if backend == "flash_attention_3":
        if version not in (None, 3):
            raise ValueError("flash_attention_3 cannot be combined with a different attention version")
        require_flash_attention_3(device, dropout_p=dropout_p, window_size=window_size)
        return 3
    if backend != "flash_attention":
        raise ValueError(f"Unsupported Wan attention backend: {backend!r}")
    fa3_eligible = FLASH_ATTN_3_AVAILABLE and is_hopper_gpu(device) and dropout_p == 0.0 and tuple(window_size) == (-1, -1)
    if version == 3 and not fa3_eligible:
        warnings.warn("Flash attention 3 is not available for this device/options, use flash attention 2 instead.")
    return 3 if version in (None, 3) and fa3_eligible else 2


def sdpa_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
):
    """SDPA in the value activation dtype, with no implicit FP32-to-half cast.

    Inputs/outputs use Wan's [batch, sequence, heads, channels] layout. Like
    FlashAttention, causal/local masks are bottom-right aligned when query and
    key lengths differ. Slice padding per sample rather than allocating a large
    dense padding mask. Query padding is returned as zeros.

    Q/K normalization may promote half activations to FP32; align those to V's
    activation dtype. When Q/K/V are FP32 (running_dtype=fp32), they stay FP32,
    even inside an outer autocast context.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Wan attention expects four-dimensional Q/K/V")
    if q.shape[0] != k.shape[0] or k.shape[:3] != v.shape[:3] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Incompatible Wan attention Q/K/V shapes")
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError("Query heads must be divisible by key/value heads")
    if len(window_size) != 2 or any(int(value) != value or value < -1 for value in window_size):
        raise ValueError("window_size must contain two integers >= -1")

    def lengths(value, maximum):
        if value is None:
            return [maximum] * q.shape[0]
        result = torch.as_tensor(value).tolist()
        if not isinstance(result, list) or len(result) != q.shape[0] or any(int(n) != n or not 0 <= n <= maximum for n in result):
            raise ValueError("Attention lengths must have one valid integer per batch sample")
        return [int(n) for n in result]

    query_lengths = lengths(q_lens, q.shape[1])
    key_lengths = lengths(k_lens, k.shape[1])
    out_dtype = q.dtype

    def attend(query, key, value):
        lq, lk = query.shape[1], key.shape[1]
        if lq == 0 or lk == 0:
            # Retain zero gradients for all three inputs in empty sequences.
            return query.new_zeros((*query.shape[:3], value.shape[-1])) + (query.sum() + key.sum() + value.sum()) * 0
        query = query.transpose(1, 2).to(value.dtype)
        key = key.transpose(1, 2).to(value.dtype)
        value = value.transpose(1, 2)
        if q_scale is not None:
            query = query * q_scale
        mask = None
        use_causal = causal and lq == lk and tuple(window_size) == (-1, -1)
        if tuple(window_size) != (-1, -1) or (causal and not use_causal):
            query_positions = torch.arange(lq, device=query.device)[:, None] + lk - lq
            key_positions = torch.arange(lk, device=query.device)[None, :]
            mask = torch.ones((lq, lk), device=query.device, dtype=torch.bool)
            if causal:
                mask &= key_positions <= query_positions
            if window_size[0] >= 0:
                mask &= key_positions >= query_positions - window_size[0]
            if window_size[1] >= 0:
                mask &= key_positions <= query_positions + window_size[1]
        with torch.autocast(device_type=query.device.type, enabled=False):
            output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=use_causal,
                scale=softmax_scale,
                enable_gqa=query.shape[1] != key.shape[1],
            )
        return output.transpose(1, 2).to(out_dtype)

    if all(n == q.shape[1] for n in query_lengths) and all(n == k.shape[1] for n in key_lengths):
        return attend(q, k, v).contiguous()
    outputs = []
    for index, (lq, lk) in enumerate(zip(query_lengths, key_lengths)):
        output = attend(q[index : index + 1, :lq], k[index : index + 1, :lk], v[index : index + 1, :lk])
        outputs.append(torch.nn.functional.pad(output, (0, 0, 0, 0, 0, q.shape[1] - lq)))
    return torch.cat(outputs, dim=0).contiguous()


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
    backend="flash_attention",
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    backend:        'flash_attention' (legacy auto), strict 'flash_attention_3',
                    or dtype-preserving 'sdpa'.
    """
    if backend == "sdpa":
        return sdpa_attention(
            q,
            k,
            v,
            q_lens,
            k_lens,
            dropout_p,
            softmax_scale,
            q_scale,
            causal,
            window_size,
        )
    resolved_version = resolve_flash_attention_version(backend, version, q.device, dropout_p=dropout_p, window_size=window_size)
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    # apply attention
    if resolved_version == 3:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or (FLASH_ATTN_3_AVAILABLE and is_hopper_gpu(q.device)):
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn("Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.")
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
