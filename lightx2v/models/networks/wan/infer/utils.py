import torch
import torch.distributed as dist

from lightx2v.common.ops.rope import RopeTemplate, TorchComplexRope
from lightx2v.utils.envs import *
from lightx2v.utils.registry_factory import ROPE_REGISTER


@ROPE_REGISTER("wan_causal_rope")
class WanCausalRope(RopeTemplate):
    def __init__(self, layout="interleaved", compute_dtype=torch.float64):
        super().__init__(layout=layout, compute_dtype=compute_dtype)
        if layout != "interleaved":
            raise ValueError("WanCausalRope only supports interleaved layout.")
        self.torch_rope = TorchComplexRope(compute_dtype=compute_dtype)

    def apply(self, q, k, freqs, grid_sizes=None, start_frame=0, **kwargs):
        if grid_sizes is None:
            return self.torch_rope.apply(q, k, freqs, **kwargs)
        return (
            self.apply_single(q, freqs, grid_sizes=grid_sizes, start_frame=start_frame),
            self.apply_single(k, freqs, grid_sizes=grid_sizes, start_frame=start_frame),
        )

    def apply_single(self, x, freqs, grid_sizes=None, start_frame=0, **kwargs):
        if grid_sizes is None:
            return self.torch_rope.apply_single(x, freqs, **kwargs)
        if x.is_cuda:
            from lightx2v.models.networks.wan.infer.triton_ops import causal_rope_apply_triton

            return causal_rope_apply_triton(x, grid_sizes, freqs, start_frame=start_frame)
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
            output.append(self.torch_rope.apply_single(x[i], pos_freqs))
        return torch.stack(output).type_as(x)

    def apply_audio_cache(
        self,
        x,
        freqs,
        *,
        h,
        w,
        token_start,
        token_end,
        ref_tokens,
        local_per_frame,
        world_size=1,
        rank=0,
        global_end=None,
        sink_tokens=0,
    ):
        if x.is_cuda:
            from lightx2v.models.networks.wan.infer.triton_ops import apply_audio_cache_rope

            return apply_audio_cache_rope(
                x,
                freqs,
                h=h,
                w=w,
                token_start=token_start,
                ref_tokens=ref_tokens,
                local_per_frame=local_per_frame,
                world_size=world_size,
                rank=rank,
                global_end=global_end,
                sink_tokens=sink_tokens,
            ).to(x.dtype)

        c = x.size(-1) // 2
        temporal_dim = c - 2 * (c // 3)
        freq_parts = freqs.split([temporal_dim, c // 3, c // 3], dim=1)
        spatial = torch.cat(
            [
                freq_parts[1][:h].view(h, 1, -1).expand(h, w, -1),
                freq_parts[2][:w].view(1, w, -1).expand(h, w, -1),
            ],
            dim=-1,
        ).reshape(h * w, -1)
        if world_size > 1:
            padding = (world_size - spatial.size(0) % world_size) % world_size
            if padding:
                spatial = torch.cat([spatial, torch.ones(padding, spatial.size(1), dtype=spatial.dtype, device=spatial.device)], dim=0)
            spatial = torch.chunk(spatial, world_size, dim=0)[rank][:local_per_frame]

        positions = torch.arange(token_start, token_end, device=freqs.device, dtype=torch.long)
        if global_end is not None:
            recent_local = max(0, token_end - int(sink_tokens))
            recent_start = int(global_end) - recent_local
            recent = recent_start + positions - int(sink_tokens)
            positions = torch.where(positions < int(sink_tokens), positions, recent)
        is_ref = positions < ref_tokens
        gen_idx = torch.clamp(positions - ref_tokens, min=0)
        ref_frames = ref_tokens // local_per_frame
        frame_idx = torch.where(is_ref, positions // local_per_frame, ref_frames + gen_idx // local_per_frame)
        spatial_idx = torch.where(is_ref, positions % local_per_frame, gen_idx % local_per_frame)
        pos_freqs = torch.cat([freq_parts[0][frame_idx], spatial[spatial_idx]], dim=-1).unsqueeze(1)
        return self.torch_rope.apply_single(x, pos_freqs.to(torch.complex64)).to(x.dtype)


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
