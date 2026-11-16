"""BF16 clean-history writes and attention KV reads with split-half RoPE."""

import torch
import triton
import triton.language as tl

from .fp8_communication import FP8QKV, fp8_packet_layout


@triton.jit
def _gather_rope_kv(
    prefix_k,
    prefix_v,
    prefix_lengths: tl.constexpr,
    query,
    output_query,
    current_k,
    current_v,
    fp8_payload,
    fp8_scales,
    fp8_layout: tl.constexpr,
    stored_k,
    stored_v,
    output_k,
    output_v,
    cosine,
    sine,
    current_rows: tl.constexpr,
    prefix_rows: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    q_strides: tl.constexpr,
    k_strides: tl.constexpr,
    v_strides: tl.constexpr,
    freq_stride: tl.constexpr,
    read_prefix: tl.constexpr,
    persist: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    tokens = rows // heads + (0 if read_prefix else prefix_rows)
    head = rows % heads
    cols = tl.arange(0, block_d)
    valid = (tokens[:, None] < prefix_rows + current_rows) & (cols[None, :] < dim)
    current = (tokens[:, None] >= prefix_rows) & valid
    local = tokens - prefix_rows
    partner = tl.where(cols < rotary_dim // 2, cols + rotary_dim // 2, cols - rotary_dim // 2)
    rotating = cols < rotary_dim

    if fp8_layout is not None:
        local_rows: tl.constexpr = fp8_layout[0]
        scale_offset: tl.constexpr = fp8_layout[1]
        packet_stride: tl.constexpr = fp8_layout[2]
        peer = local // local_rows
        peer_token = local % local_rows
        q_row = peer_token * 3 * heads + head
        k_row = q_row + heads
        v_row = k_row + heads
        payload_base = peer[:, None] * packet_stride
        scale_base = peer * (packet_stride // 4) + scale_offset
        current_row = (local >= 0) & (local < current_rows)
        k_scale = tl.load(fp8_scales + scale_base + k_row, current_row, 0)[:, None]
        v_scale = tl.load(fp8_scales + scale_base + v_row, current_row, 0)[:, None]
        key_offset = payload_base + k_row[:, None] * dim
        value_offset = payload_base + v_row[:, None] * dim
        key = tl.load(fp8_payload + key_offset + cols[None, :], current, 0.0).to(tl.float32)
        other = tl.load(fp8_payload + key_offset + partner[None, :], current & rotating[None, :], 0.0).to(tl.float32)
        value = tl.load(fp8_payload + value_offset + cols[None, :], current, 0.0).to(tl.float32)
        # Preserve the old unpack's BF16 rounding before RoPE or raw-KV writes.
        key = (key * k_scale + 0.0).to(tl.bfloat16).to(tl.float32)
        other = (other * k_scale + 0.0).to(tl.bfloat16).to(tl.float32)
        value = (value * v_scale + 0.0).to(tl.bfloat16)
    else:
        key_offset = local[:, None] * k_strides[0] + head[:, None] * k_strides[1]
        value_offset = local[:, None] * v_strides[0] + head[:, None] * v_strides[1]
        key = tl.load(current_k + key_offset + cols[None, :] * k_strides[2], current, 0).to(tl.float32)
        other = tl.load(current_k + key_offset + partner[None, :] * k_strides[2], current & rotating[None, :], 0).to(tl.float32)
        value = tl.load(current_v + value_offset + cols[None, :] * v_strides[2], current, 0)
    if persist:
        destination = (local[:, None] * heads + head[:, None]) * dim + cols[None, :]
        tl.store(stored_k + destination, key, current)
        tl.store(stored_v + destination, value, current)

    if read_prefix:
        begin = 0
        for index in tl.static_range(len(prefix_lengths)):
            length = prefix_lengths[index]
            selected = (tokens[:, None] >= begin) & (tokens[:, None] < begin + length) & valid
            offset = ((tokens[:, None] - begin) * heads + head[:, None]) * dim
            key = tl.where(selected, tl.load(prefix_k[index] + offset + cols[None, :], selected, 0).to(tl.float32), key)
            other = tl.where(selected, tl.load(prefix_k[index] + offset + partner[None, :], selected & rotating[None, :], 0).to(tl.float32), other)
            value = tl.where(selected, tl.load(prefix_v[index] + offset + cols[None, :], selected, 0), value)
            begin += length

    frequency = tokens[:, None] * freq_stride + cols[None, :]
    cos = tl.load(cosine + frequency, valid & rotating[None, :], 0).to(tl.bfloat16).to(tl.float32)
    sin = tl.load(sine + frequency, valid & rotating[None, :], 0).to(tl.bfloat16).to(tl.float32)
    # Match eager TorchRealRope(BF16): round each
    # multiplication before addition, and never contract them into an FMA.
    first = (key * cos).to(tl.bfloat16).to(tl.float32)
    second = (tl.where(cols[None, :] < rotary_dim // 2, -other, other) * sin).to(tl.bfloat16).to(tl.float32)
    rotated = tl.where(rotating[None, :], first + second, key)
    # Query rows have exactly the positions of the current KV suffix.
    if fp8_layout is not None:
        q_scale = tl.load(fp8_scales + scale_base + q_row, current_row, 0)[:, None]
        query_offset = payload_base + q_row[:, None] * dim
        q = tl.load(fp8_payload + query_offset + cols[None, :], current, 0.0).to(tl.float32)
        q_other = tl.load(fp8_payload + query_offset + partner[None, :], current & rotating[None, :], 0.0).to(tl.float32)
        q = (q * q_scale + 0.0).to(tl.bfloat16).to(tl.float32)
        q_other = (q_other * q_scale + 0.0).to(tl.bfloat16).to(tl.float32)
    else:
        query_offset = local[:, None] * q_strides[0] + head[:, None] * q_strides[1]
        q = tl.load(query + query_offset + cols[None, :] * q_strides[2], current, 0).to(tl.float32)
        q_other = tl.load(query + query_offset + partner[None, :] * q_strides[2], current & rotating[None, :], 0).to(tl.float32)
    q_first = (q * cos).to(tl.bfloat16).to(tl.float32)
    q_second = (tl.where(cols[None, :] < rotary_dim // 2, -q_other, q_other) * sin).to(tl.bfloat16).to(tl.float32)
    rotated_q = tl.where(rotating[None, :], q_first + q_second, q)
    query_destination = (local[:, None] * heads + head[:, None]) * dim + cols[None, :]
    tl.store(output_query + query_destination, rotated_q, current)

    destination = (tokens[:, None] * heads + head[:, None]) * dim + cols[None, :]
    tl.store(output_k + destination, rotated, valid)
    tl.store(output_v + destination, value, valid)


def gather_rope_qkv(prefix, qkv, stored, workspace, freqs, *, read_prefix):
    """Rotate queries while refreshing clean KV; read the prefix once per stage."""
    keys, values = tuple(item[0] for item in prefix), tuple(item[1] for item in prefix)
    lengths = tuple(key.shape[0] for key in keys)
    start = sum(lengths)
    if isinstance(qkv, FP8QKV):
        rows, heads, dim = qkv.shape
        local_rows = (rows + qkv.data.shape[0] - 1) // qkv.data.shape[0]
        scale_offset, stride = fp8_packet_layout(local_rows * 3 * heads, dim)
        fp8_layout = (local_rows, scale_offset, stride)
        payload, scales = qkv.data.view(torch.float8_e4m3fn), qkv.data.view(torch.float32)
        q = k = v = None
        q_strides = k_strides = v_strides = (0, 0, 0)
    else:
        q, k, v = qkv
        rows, heads, dim = k.shape
        payload = scales = fp8_layout = None
        q_strides, k_strides, v_strides = q.stride(), k.stride(), v.stride()
    rotary_dim = freqs[0].shape[-1]
    if rotary_dim % 2 or rotary_dim > dim:
        raise ValueError("CleanAR RoPE width must be even and fit the attention head")
    write_k, write_v = stored if stored is not None else (None, None)
    count = (rows + start if read_prefix else rows) * heads
    rotated_q = torch.empty((rows, heads, dim), device=workspace[0].device, dtype=workspace[0].dtype)
    _gather_rope_kv[(triton.cdiv(count, 8),)](
        keys,
        values,
        lengths,
        q,
        rotated_q,
        k,
        v,
        payload,
        scales,
        fp8_layout,
        write_k,
        write_v,
        *workspace,
        *freqs,
        rows,
        start,
        heads,
        dim,
        rotary_dim,
        q_strides,
        k_strides,
        v_strides,
        freqs[0].stride(0),
        read_prefix,
        stored is not None,
        8,
        triton.next_power_of_2(dim),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return rotated_q
