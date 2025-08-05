import torch
import torch.distributed as dist
from loguru import logger
from lightx2v.utils.envs import *


def compute_freqs(c, grid_sizes, freqs):
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid_sizes[0].tolist()
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


def compute_freqs_audio(c, grid_sizes, freqs):
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid_sizes[0].tolist()
    f = f + 1  ##for r2v add 1 channel
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


def compute_freqs_audio_dist(s, c, grid_sizes, freqs):
    world_size = dist.get_world_size()
    cur_rank = dist.get_rank()
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid_sizes[0].tolist()
    f = f + 1
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
    f, h, w = grid_sizes[0].tolist()
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


def compute_freqs_dist(s, c, grid_sizes, freqs):
    world_size = dist.get_world_size()
    cur_rank = dist.get_rank()
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid_sizes[0].tolist()
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


def apply_rotary_emb(x, freqs_i):
    n = x.size(1)
    seq_len = freqs_i.size(0)

    x_i = torch.view_as_complex(x[:seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
    # Apply rotary embedding
    x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
    x_i = torch.cat([x_i, x[seq_len:]])
    return x_i.to(torch.bfloat16)


def apply_rotary_emb_chunk(x, freqs_i, chunk_size, remaining_chunk_size=100):
    n = x.size(1)
    seq_len = freqs_i.size(0)

    output_chunks = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        x_chunk = x[start:end]
        freqs_chunk = freqs_i[start:end]

        x_chunk_complex = torch.view_as_complex(x_chunk.to(torch.float32).reshape(end - start, n, -1, 2))
        x_chunk_embedded = torch.view_as_real(x_chunk_complex * freqs_chunk).flatten(2).to(torch.bfloat16)
        output_chunks.append(x_chunk_embedded)
        del x_chunk_complex, x_chunk_embedded
        torch.cuda.empty_cache()

    result = []
    for chunk in output_chunks:
        result.append(chunk)
    del output_chunks
    torch.cuda.empty_cache()

    for start in range(seq_len, x.size(0), remaining_chunk_size):
        end = min(start + remaining_chunk_size, x.size(0))
        result.append(x[start:end])

    x_i = torch.cat(result, dim=0)
    del result
    torch.cuda.empty_cache()

    return x_i.to(torch.bfloat16)


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    if GET_DTYPE() == "BF16":
        x = x.to(torch.bfloat16)
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
    # w = torch.round(w)
    # w = torch.clamp(w, min=cfg_min, max=cfg_max)
    w = (w - cfg_min) / (cfg_max - cfg_min)  # [0, 1]
    w = w * target_range
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype).to(w.device) * -emb).to(w.device)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1).to(w.device))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb
