import triton
import triton.language as tl


@triton.jit
def _rotate_pair(real, imag, cosine, sine):
    real, imag = real.to(tl.float64), imag.to(tl.float64)
    return (real * cosine - imag * sine).to(tl.float32), (real * sine + imag * cosine).to(tl.float32)


@triton.jit
def causal_rope_kernel(
    query,
    key,
    query_out,
    key_out,
    freqs,
    grid_sizes,
    start_frame,
    pairs: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    with_query: tl.constexpr,
    block_pairs: tl.constexpr,
):
    offsets = tl.program_id(0) * block_pairs + tl.arange(0, block_pairs)
    mask = offsets < pairs
    half_dim: tl.constexpr = dim // 2
    spatial_dim: tl.constexpr = half_dim // 3
    temporal_dim: tl.constexpr = half_dim - 2 * spatial_dim
    token = offsets // (heads * half_dim)
    feature = offsets % half_dim
    height, width = tl.load(grid_sizes + 1), tl.load(grid_sizes + 2)
    frame = token // (height * width)
    row = token // width % height
    column = token % width
    position = tl.where(feature < temporal_dim, tl.load(start_frame) + frame, tl.where(feature < temporal_dim + spatial_dim, row, column))
    address = (position * half_dim + feature) * 2
    cosine = tl.load(freqs + address, mask, 0).to(tl.float64)
    sine = tl.load(freqs + address + 1, mask, 0).to(tl.float64)
    if with_query:
        query_real = tl.load(query + offsets * 2, mask, 0)
        query_imag = tl.load(query + offsets * 2 + 1, mask, 0)
        real, imag = _rotate_pair(query_real, query_imag, cosine, sine)
        tl.store(query_out + offsets * 2, real, mask)
        tl.store(query_out + offsets * 2 + 1, imag, mask)
    key_real = tl.load(key + offsets * 2, mask, 0)
    key_imag = tl.load(key + offsets * 2 + 1, mask, 0)
    real, imag = _rotate_pair(key_real, key_imag, cosine, sine)
    tl.store(key_out + offsets * 2, real, mask)
    tl.store(key_out + offsets * 2 + 1, imag, mask)


@triton.jit
def update_sink_key_kernel(
    current_key,
    raw_sink_key,
    key_ring,
    valid,
    freqs,
    grid_sizes,
    start_frame,
    pairs: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    window_frames: tl.constexpr,
    sink_parts: tl.constexpr,
    fixed_sink: tl.constexpr,
    quantized: tl.constexpr,
    block_pairs: tl.constexpr,
):
    position = tl.load(start_frame)
    offsets = tl.program_id(0) * block_pairs + tl.arange(0, block_pairs)
    mask = offsets < pairs
    if position == 0:
        real = tl.load(current_key + offsets * 2, mask, 0)
        imag = tl.load(current_key + offsets * 2 + 1, mask, 0)
        tl.store(raw_sink_key + offsets * 2, real, mask)
        tl.store(raw_sink_key + offsets * 2 + 1, imag, mask)
    if fixed_sink and position > window_frames:
        half_dim: tl.constexpr = dim // 2
        spatial_dim: tl.constexpr = half_dim // 3
        temporal_dim: tl.constexpr = half_dim - 2 * spatial_dim
        token = offsets // (heads * half_dim)
        feature = offsets % half_dim
        height, width = tl.load(grid_sizes + 1), tl.load(grid_sizes + 2)
        frame = token // (height * width)
        row = token // width % height
        column = token % width
        frequency_position = tl.where(feature < temporal_dim, position - window_frames + frame, tl.where(feature < temporal_dim + spatial_dim, row, column))
        address = (frequency_position * half_dim + feature) * 2
        cosine = tl.load(freqs + address, mask, 0).to(tl.float64)
        sine = tl.load(freqs + address + 1, mask, 0).to(tl.float64)
        real = tl.load(raw_sink_key + offsets * 2, mask, 0)
        imag = tl.load(raw_sink_key + offsets * 2 + 1, mask, 0)
        rotated_real, rotated_imag = _rotate_pair(real, imag, cosine, sine)
        tl.store(key_ring + offsets * 2, rotated_real, mask)
        tl.store(key_ring + offsets * 2 + 1, rotated_imag, mask)
        if quantized:
            tl.store(valid + offsets, 0, offsets < sink_parts)
