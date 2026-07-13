import torch
import torch.distributed as dist

from lightx2v.common.ops.rope import ChunkedRope, FlashInferRope, TorchComplexRope, TorchRealRope
from lightx2v.utils.envs import *

_WAN_TORCH_ROPE = TorchComplexRope()
_WAN_CAUSAL_TORCH_ROPE = TorchComplexRope(compute_dtype=torch.float64)
_WAN_TORCH_NAIVE_ROPE = TorchRealRope(layout="interleaved")
_WAN_FLASHINFER_ROPE = FlashInferRope(layout="interleaved")


def apply_wan_rope_with_torch(xq, xk, cos_sin_cache):
    return _WAN_TORCH_ROPE.apply(xq, xk, cos_sin_cache)


def apply_wan_rope_with_torch_naive(xq, xk, cos_sin_cache):
    return _WAN_TORCH_NAIVE_ROPE.apply(xq, xk, cos_sin_cache)


def apply_wan_rope_with_chunk(xq, xk, cos_sin_cache, chunk_size, rope_func):
    class _CallableRope:
        layout = "interleaved"
        compute_dtype = torch.float32

        def apply(self, q, k, freqs, **kwargs):
            return rope_func(q, k, freqs)

    return ChunkedRope(_CallableRope(), chunk_size).apply(xq, xk, cos_sin_cache)


def apply_wan_rope_with_flashinfer(xq, xk, cos_sin_cache):
    return _WAN_FLASHINFER_ROPE.apply(xq, xk, cos_sin_cache)


def compute_freqs(c, grid_sizes, freqs):
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid_sizes
    seq_len = f * h * w
    freqs_i = torch.cat(
        [
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(seq_len, 1, -1)

    return freqs_i


def compute_freqs_dist(s, c, grid_sizes, freqs, seq_p_group):
    world_size = dist.get_world_size(seq_p_group)
    cur_rank = dist.get_rank(seq_p_group)
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid_sizes
    seq_len = f * h * w
    freqs_i = torch.cat(
        [
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(seq_len, 1, -1)

    freqs_i = pad_freqs(freqs_i, s * world_size)
    s_per_rank = s
    freqs_i_rank = freqs_i[(cur_rank * s_per_rank) : ((cur_rank + 1) * s_per_rank), :, :]
    return freqs_i_rank


def compute_freqs_causvid(c, grid_sizes, freqs, start_frame=0):
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid_sizes
    seq_len = f * h * w
    freqs_i = torch.cat(
        [
            freqs[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(seq_len, 1, -1)

    return freqs_i


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(pad_size, s1, s2, dtype=original_tensor.dtype, device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


def apply_rotary_emb(x, freqs_i):
    return _WAN_TORCH_ROPE.apply_single(x, freqs_i).to(GET_DTYPE())


def apply_rotary_emb_chunk(x, freqs_i, chunk_size, remaining_chunk_size=100):
    del remaining_chunk_size
    return ChunkedRope(_WAN_TORCH_ROPE, chunk_size).apply_single(x, freqs_i).to(GET_DTYPE())


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    x = x.to(GET_SENSITIVE_DTYPE())
    return x


def guidance_scale_embedding(w, embedding_dim=256, cfg_range=(1.0, 6.0), target_range=1000.0, dtype=torch.float32):
    """
    Args:
    timesteps: torch.Tensor: generate embedding vectors at these timesteps
    embedding_dim: int: dimension of the embeddings to generate
    dtype: data type of the generated embeddings

    Returns:
    embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    cfg_min, cfg_max = cfg_range
    w = torch.round(w)
    w = torch.clamp(w, min=cfg_min, max=cfg_max)
    w = (w - cfg_min) / (cfg_max - cfg_min)  # [0, 1]
    w = w * target_range
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype).to(w.device) * -emb).to(w.device)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1).to(w.device))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    c = x.size(3) // 2
    freq_parts = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (frames, height, width) in enumerate(grid_sizes.tolist()):
        seq_len = frames * height * width
        pos_freqs = torch.cat(
            [
                freq_parts[0][start_frame : start_frame + frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                freq_parts[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                freq_parts[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        output.append(_WAN_CAUSAL_TORCH_ROPE.apply_single(x[i], pos_freqs))
    return torch.stack(output).type_as(x)
