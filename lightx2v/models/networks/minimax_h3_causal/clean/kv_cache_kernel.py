"""Clean-history writes and attention KV reads with cached V layouts."""

import torch
import triton
import triton.language as tl

from .fp8_communication import FP8QKV, fp8_packet_layout

# Use the encoder's packet layout calculation in the GPU reader as well.
_fp8_packet_layout = triton.jit(fp8_packet_layout)


@triton.jit(do_not_specialize=["current_rows", "prefix_rows"])
def _gather_rope_kv(
    prefix_k,
    prefix_v,
    prefix_lengths,
    query,
    output_query,
    current_k,
    current_v,
    fp8_payload,
    fp8_scales,
    fp8_peers: tl.constexpr,
    stored_k,
    stored_v,
    output_k,
    output_v,
    transposed_v: tl.constexpr,
    cosine,
    sine,
    current_rows,
    prefix_rows,
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

    if fp8_payload is not None:
        local_rows = tl.cdiv(current_rows, fp8_peers)
        scale_offset, packet_stride = _fp8_packet_layout(local_rows * 3 * heads, dim)
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
        # Use the same BF16 keys for current attention and later cache reads.
        key = (key * k_scale).to(tl.bfloat16).to(tl.float32)
        other = (other * k_scale).to(tl.bfloat16).to(tl.float32)
        value = (value * v_scale).to(tl.bfloat16)
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
        for index in tl.static_range(len(prefix_k)):
            length = tl.load(prefix_lengths + index)
            selected = (tokens[:, None] >= begin) & (tokens[:, None] < begin + length) & valid
            offset = ((tokens[:, None] - begin) * heads + head[:, None]) * dim
            key = tl.where(selected, tl.load(prefix_k[index] + offset + cols[None, :], selected, 0).to(tl.float32), key)
            other = tl.where(selected, tl.load(prefix_k[index] + offset + partner[None, :], selected & rotating[None, :], 0).to(tl.float32), other)
            value = tl.where(selected, tl.load(prefix_v[index] + offset + cols[None, :], selected, 0), value)
            begin += length

    frequency = tokens[:, None] * freq_stride + cols[None, :]
    cos = tl.load(cosine + frequency, valid & rotating[None, :], 0).to(tl.bfloat16).to(tl.float32)
    sin = tl.load(sine + frequency, valid & rotating[None, :], 0).to(tl.bfloat16).to(tl.float32)
    # Accumulate RoPE in FP32 and round when storing the BF16 output.
    rotated = key * cos + tl.where(cols[None, :] < rotary_dim // 2, -other, other) * sin
    rotated = tl.where(rotating[None, :], rotated, key)
    # Query rows have exactly the positions of the current KV suffix.
    if fp8_payload is not None:
        q_scale = tl.load(fp8_scales + scale_base + q_row, current_row, 0)[:, None]
        query_offset = payload_base + q_row[:, None] * dim
        q = tl.load(fp8_payload + query_offset + cols[None, :], current, 0.0).to(tl.float32)
        q_other = tl.load(fp8_payload + query_offset + partner[None, :], current & rotating[None, :], 0.0).to(tl.float32)
        q = (q * q_scale).to(tl.bfloat16).to(tl.float32)
        q_other = (q_other * q_scale).to(tl.bfloat16).to(tl.float32)
    else:
        query_offset = local[:, None] * q_strides[0] + head[:, None] * q_strides[1]
        q = tl.load(query + query_offset + cols[None, :] * q_strides[2], current, 0).to(tl.float32)
        q_other = tl.load(query + query_offset + partner[None, :] * q_strides[2], current & rotating[None, :], 0).to(tl.float32)
    rotated_q = q * cos + tl.where(cols[None, :] < rotary_dim // 2, -q_other, q_other) * sin
    rotated_q = tl.where(rotating[None, :], rotated_q, q)
    query_destination = (local[:, None] * heads + head[:, None]) * dim + cols[None, :]
    tl.store(output_query + query_destination, rotated_q, current)

    destination = (tokens[:, None] * heads + head[:, None]) * dim + cols[None, :]
    tl.store(output_k + destination, rotated, valid)
    if not transposed_v:
        tl.store(output_v + destination, value, valid)


@triton.jit(do_not_specialize=["current_rows", "prefix_rows", "output_v_span"])
def _gather_v(
    prefix_v,
    prefix_lengths,
    current_v,
    fp8_payload,
    fp8_scales,
    fp8_peers: tl.constexpr,
    output_v,
    current_rows,
    prefix_rows,
    heads: tl.constexpr,
    dim: tl.constexpr,
    v_strides: tl.constexpr,
    output_v_span,
    read_prefix: tl.constexpr,
):
    # Transpose and permute only the changed suffix between denoising steps.
    tokens = tl.program_id(0) * 64 + tl.arange(0, 64) + (0 if read_prefix else prefix_rows)
    head = tl.program_id(1)
    cols = tl.program_id(2) * 32 + tl.arange(0, 32)
    valid = (tokens[:, None] < prefix_rows + current_rows) & (cols[None, :] < dim)
    local = tokens - prefix_rows
    current = (local[:, None] >= 0) & valid
    if fp8_payload is not None:
        local_rows = tl.cdiv(current_rows, fp8_peers)
        scale_offset, packet_stride = _fp8_packet_layout(local_rows * 3 * heads, dim)
        peer = local // local_rows
        v_row = (local % local_rows) * 3 * heads + 2 * heads + head
        scale = tl.load(fp8_scales + peer * (packet_stride // 4) + scale_offset + v_row, (local >= 0) & (local < current_rows), 0)
        value = tl.load(fp8_payload + peer[:, None] * packet_stride + v_row[:, None] * dim + cols[None, :], current, 0.0).to(tl.float32)
        value = (value * scale[:, None]).to(tl.bfloat16)
    else:
        value = tl.load(current_v + local[:, None] * v_strides[0] + head * v_strides[1] + cols[None, :] * v_strides[2], current, 0)
    if read_prefix:
        begin = 0
        for index in tl.static_range(len(prefix_v)):
            length = tl.load(prefix_lengths + index)
            selected = (tokens[:, None] >= begin) & (tokens[:, None] < begin + length) & valid
            offset = ((tokens[:, None] - begin) * heads + head) * dim + cols[None, :]
            value = tl.where(selected, tl.load(prefix_v[index] + offset, selected, 0), value)
            begin += length
    permuted = tokens // 16 * 16 + (tokens % 16 // 8) * 2 + (tokens % 8 // 2) * 4 + tokens % 2
    offset = (cols[None, :] * heads + head) * output_v_span + permuted[:, None]
    padded_end = tl.cdiv(prefix_rows + current_rows, 64) * 64
    tl.store(output_v + offset, tl.where(valid, value, 0), (tokens[:, None] < padded_end) & (cols[None, :] < dim))


def gather_transposed_v(prefix, v, workspace, *, prefix_lengths, read_prefix):
    """Refresh V independently of RoPE, including FP16 and unfused RoPE paths."""
    lengths = tuple(value.shape[0] for value in prefix)
    rows, heads, dim = v.shape
    start = sum(lengths)
    end = triton.cdiv(start + rows, 64) * 64
    count = end if read_prefix else end - start
    _gather_v[(triton.cdiv(count, 64), heads, triton.cdiv(dim, 32))](
        prefix,
        prefix_lengths,
        v,
        None,
        None,
        0,
        workspace,
        rows,
        start,
        heads,
        dim,
        v.stride(),
        workspace.shape[-1],
        read_prefix,
    )


def gather_rope_qkv(prefix, qkv, stored, workspace, freqs, *, prefix_lengths, read_prefix):
    """Rotate queries while refreshing clean KV; read the prefix once per stage."""
    keys, values = tuple(item[0] for item in prefix), tuple(item[1] for item in prefix)
    lengths = tuple(key.shape[0] for key in keys)
    start = sum(lengths)
    if isinstance(qkv, FP8QKV):
        rows, heads, dim = qkv.shape
        fp8_peers = qkv.data.shape[0]
        payload, scales = qkv.data.view(torch.float8_e4m3fn), qkv.data.view(torch.float32)
        q = k = v = None
        q_strides = k_strides = v_strides = (0, 0, 0)
    else:
        q, k, v = qkv
        rows, heads, dim = k.shape
        payload = scales = None
        fp8_peers = 0
        q_strides, k_strides, v_strides = q.stride(), k.stride(), v.stride()
    rotary_dim = freqs[0].shape[-1]
    if rotary_dim % 2 or rotary_dim > dim:
        raise ValueError("CleanAR RoPE width must be even and fit the attention head")
    write_k, write_v = stored if stored is not None else (None, None)
    transposed_v = workspace[1].ndim == 4
    end = triton.cdiv(rows + start, 64) * 64 if transposed_v else rows + start
    count = (end if read_prefix else end - start) * heads
    rotated_q = torch.empty((rows, heads, dim), device=workspace[0].device, dtype=workspace[0].dtype)
    _gather_rope_kv[(triton.cdiv(count, 8),)](
        keys,
        values,
        prefix_lengths,
        q,
        rotated_q,
        k,
        v,
        payload,
        scales,
        fp8_peers,
        write_k,
        write_v,
        *workspace,
        transposed_v,
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
    )
    if transposed_v:
        _gather_v[(triton.cdiv(count // heads, 64), heads, triton.cdiv(dim, 32))](
            values,
            prefix_lengths,
            v,
            payload,
            scales,
            fp8_peers,
            workspace[1],
            rows,
            start,
            heads,
            dim,
            v_strides,
            workspace[1].shape[-1],
            read_prefix,
        )
    return rotated_q
