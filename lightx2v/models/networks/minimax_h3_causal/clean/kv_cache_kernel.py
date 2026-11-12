"""BF16 clean-history writes and attention KV reads with split-half RoPE."""

import triton
import triton.language as tl


@triton.jit
def _gather_rope_kv(
    prefix_k,
    prefix_v,
    prefix_lengths: tl.constexpr,
    query,
    output_query,
    current_k,
    current_v,
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


def gather_rope_qkv(prefix, q, k, v, stored, workspace, freqs, *, read_prefix):
    """Rotate queries while refreshing clean KV; read the prefix once per stage."""
    keys, values = tuple(item[0] for item in prefix), tuple(item[1] for item in prefix)
    lengths = tuple(key.shape[0] for key in keys)
    start = sum(lengths)
    rows, heads, dim = k.shape
    rotary_dim = freqs[0].shape[-1]
    if rotary_dim % 2 or rotary_dim > dim:
        raise ValueError("CleanAR RoPE width must be even and fit the attention head")
    write_k, write_v = stored if stored is not None else (k, v)
    count = (rows + start if read_prefix else rows) * heads
    rotated_q = q.new_empty(q.shape)
    _gather_rope_kv[(triton.cdiv(count, 8),)](
        keys,
        values,
        lengths,
        q,
        rotated_q,
        k,
        v,
        write_k,
        write_v,
        *workspace,
        *freqs,
        rows,
        start,
        heads,
        dim,
        rotary_dim,
        q.stride(),
        k.stride(),
        v.stride(),
        freqs[0].stride(0),
        read_prefix,
        stored is not None,
        8,
        triton.next_power_of_2(dim),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return rotated_q
