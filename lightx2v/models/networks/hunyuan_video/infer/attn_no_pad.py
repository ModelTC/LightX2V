import torch
from einops import rearrange
from flash_attn import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import pad_input, unpad_input

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
except ImportError:
    flash_attn_varlen_func_v3 = None
    print("flash_attn_varlen_func_v3 not available")

if torch.cuda.get_device_capability(0) in [(8, 9), (12, 0)]:
    try:
        from sageattention import sageattn_qk_int8_pv_fp16_triton as sageattn
    except ImportError:
        logger.info("sageattn not found, please install sageattention first")
        sageattn = None
else:
    try:
        from sageattention import sageattn
    except ImportError:
        logger.info("sageattn not found, please install sageattention first")
        sageattn = None

try:
    from sageattn3 import sageattn3_blackwell
except ImportError:
    logger.info("sageattn3 not found, please install sageattention first")
    sageattn3_blackwell = None


def flash_attn_no_pad(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False):
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(x, key_padding_mask)

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )
    output = rearrange(
        pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


def flash_attn_no_pad_v3(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False):
    if flash_attn_varlen_func_v3 is None:
        raise ImportError("FlashAttention V3 backend not available")

    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    query, key, value = qkv.unbind(dim=2)

    query_unpad, indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(rearrange(query, "b s h d -> b s (h d)"), key_padding_mask)
    key_unpad, _, cu_seqlens_k, _, _ = unpad_input(rearrange(key, "b s h d -> b s (h d)"), key_padding_mask)
    value_unpad, _, _, _, _ = unpad_input(rearrange(value, "b s h d -> b s (h d)"), key_padding_mask)

    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=nheads)
    key_unpad = rearrange(key_unpad, "nnz (h d) -> nnz h d", h=nheads)
    value_unpad = rearrange(value_unpad, "nnz (h d) -> nnz h d", h=nheads)

    output_unpad = flash_attn_varlen_func_v3(
        query_unpad, key_unpad, value_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_q, softmax_scale=softmax_scale, causal=causal, deterministic=deterministic
    )

    output = rearrange(pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen), "b s (h d) -> b s h d", h=nheads)
    return output


def sage_attn_no_pad_v2(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False):
    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    query, key, value = qkv.unbind(dim=2)

    query_unpad, indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(rearrange(query, "b s h d -> b s (h d)"), key_padding_mask)
    key_unpad, _, cu_seqlens_k, _, _ = unpad_input(rearrange(key, "b s h d -> b s (h d)"), key_padding_mask)
    value_unpad, _, _, _, _ = unpad_input(rearrange(value, "b s h d -> b s (h d)"), key_padding_mask)

    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=nheads)
    key_unpad = rearrange(key_unpad, "nnz (h d) -> nnz h d", h=nheads)
    value_unpad = rearrange(value_unpad, "nnz (h d) -> nnz h d", h=nheads)

    output_unpad = sageattn(
        query_unpad.unsqueeze(0),
        key_unpad.unsqueeze(0),
        value_unpad.unsqueeze(0),
        tensor_layout="NHD",
    ).squeeze(0)

    output = rearrange(pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen), "b s (h d) -> b s h d", h=nheads)
    return output


def sage_attn_no_pad_v3(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False):
    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    query, key, value = qkv.unbind(dim=2)

    query_unpad, indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(rearrange(query, "b s h d -> b s (h d)"), key_padding_mask)
    key_unpad, _, cu_seqlens_k, _, _ = unpad_input(rearrange(key, "b s h d -> b s (h d)"), key_padding_mask)
    value_unpad, _, _, _, _ = unpad_input(rearrange(value, "b s h d -> b s (h d)"), key_padding_mask)

    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=nheads)
    key_unpad = rearrange(key_unpad, "nnz (h d) -> nnz h d", h=nheads)
    value_unpad = rearrange(value_unpad, "nnz (h d) -> nnz h d", h=nheads)

    output_unpad = sageattn3_blackwell(query_unpad.unsqueeze(0).transpose(1, 2), key_unpad.unsqueeze(0).transpose(1, 2), value_unpad.unsqueeze(0).transpose(1, 2)).transpose(1, 2).squeeze(0)

    output = rearrange(pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen), "b s (h d) -> b s h d", h=nheads)
    return output
