import math

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from lightx2v_train.runtime.distributed import get_sequence_parallel_world_size
from lightx2v_train.runtime.sequence_parallel import all_to_all_4d, is_sequence_parallel_enabled

from .attention import attention
from .model import WAN_CROSSATTENTION_CLASSES, MLPProj, WanLayerNorm, WanRMSNorm, rope_params, sinusoidal_embedding_1d

# Match LongLive's sequence-parallel training path. The non-compiled
# FlexAttention helpers materialize large temporary masks on long TF sequences.
flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [freqs[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1), freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1), freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def distributed_flex_attention(roped_query, roped_key, value, block_mask):
    roped_query = all_to_all_4d(roped_query, scatter_dim=2, gather_dim=1)
    roped_key = all_to_all_4d(roped_key, scatter_dim=2, gather_dim=1)
    value = all_to_all_4d(value, scatter_dim=2, gather_dim=1)

    padded_length = math.ceil(roped_query.shape[1] / 128) * 128 - roped_query.shape[1]
    if padded_length > 0:
        pad_shape = (roped_query.shape[0], padded_length, roped_query.shape[2], roped_query.shape[3])
        roped_query = torch.cat([roped_query, roped_query.new_zeros(pad_shape)], dim=1)
        roped_key = torch.cat([roped_key, roped_key.new_zeros(pad_shape)], dim=1)
        value = torch.cat([value, value.new_zeros(pad_shape)], dim=1)

    output = flex_attention(
        query=roped_query.transpose(2, 1),
        key=roped_key.transpose(2, 1),
        value=value.transpose(2, 1),
        block_mask=block_mask,
    )
    if padded_length > 0:
        output = output[:, :, :-padded_length]
    output = output.transpose(2, 1)
    return all_to_all_4d(output, scatter_dim=1, gather_dim=2)


class CausalWanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        local_frame_offset=None,
        balanced_sequence_parallel=False,
        defer_cache_update=False,
        detach_cache_update=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        rope_start_frame = 0 if local_frame_offset is None else int(local_frame_offset)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = s == seq_lens[0].item() * 2
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = causal_rope_apply(q_chunk[ii], grid_sizes, freqs, start_frame=rope_start_frame).type_as(v)
                    rk = causal_rope_apply(k_chunk[ii], grid_sizes, freqs, start_frame=rope_start_frame).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                if balanced_sequence_parallel and is_sequence_parallel_enabled():
                    x = distributed_flex_attention(roped_query, roped_key, v, block_mask)
                else:
                    padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                    padded_roped_query = torch.cat([roped_query, torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]], device=q.device, dtype=v.dtype)], dim=1)
                    padded_roped_key = torch.cat([roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]], device=k.device, dtype=v.dtype)], dim=1)
                    padded_v = torch.cat([v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]], device=v.device, dtype=v.dtype)], dim=1)
                    x = flex_attention(query=padded_roped_query.transpose(2, 1), key=padded_roped_key.transpose(2, 1), value=padded_v.transpose(2, 1), block_mask=block_mask)
                    if padded_length > 0:
                        x = x[:, :, :-padded_length]
                    x = x.transpose(2, 1)

            else:
                roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=rope_start_frame).type_as(v)
                roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=rope_start_frame).type_as(v)

                if balanced_sequence_parallel and is_sequence_parallel_enabled():
                    x = distributed_flex_attention(roped_query, roped_key, v, block_mask)
                else:
                    padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                    padded_roped_query = torch.cat([roped_query, torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]], device=q.device, dtype=v.dtype)], dim=1)
                    padded_roped_key = torch.cat([roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]], device=k.device, dtype=v.dtype)], dim=1)
                    padded_v = torch.cat([v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]], device=v.device, dtype=v.dtype)], dim=1)
                    x = flex_attention(query=padded_roped_query.transpose(2, 1), key=padded_roped_key.transpose(2, 1), value=padded_v.transpose(2, 1), block_mask=block_mask)
                    if padded_length > 0:
                        x = x[:, :, :-padded_length]
                    x = x.transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen if local_frame_offset is None else int(local_frame_offset)
            roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            sequence_parallel_cache = bool(balanced_sequence_parallel and is_sequence_parallel_enabled())
            if sequence_parallel_cache:
                roped_query = all_to_all_4d(roped_query, scatter_dim=2, gather_dim=1)
                roped_key = all_to_all_4d(roped_key, scatter_dim=2, gather_dim=1)
                v = all_to_all_4d(v, scatter_dim=2, gather_dim=1)
                if kv_cache["k"].shape[2] != roped_key.shape[2]:
                    raise ValueError(f"Sequence-parallel KV cache stores {kv_cache['k'].shape[2]} heads, but current key has {roped_key.shape[2]} heads.")

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            attention_window_size = kv_cache_size if self.local_attn_size == -1 else self.local_attn_size * frame_seqlen
            num_new_tokens = roped_query.shape[1]
            cache_update_info = None
            detach_cache_update = bool(detach_cache_update and torch.is_grad_enabled() and (roped_key.requires_grad or v.requires_grad) and not defer_cache_update)
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                if defer_cache_update:
                    window_start = max(0, local_end_index - attention_window_size)
                    window_k_pieces = []
                    window_v_pieces = []
                    prefix_end = min(sink_tokens, local_start_index)
                    if window_start < prefix_end:
                        window_k_pieces.append(kv_cache["k"][:, window_start:prefix_end])
                        window_v_pieces.append(kv_cache["v"][:, window_start:prefix_end])
                    rolled_start = max(window_start, sink_tokens)
                    if rolled_start < local_start_index:
                        src_start = rolled_start + num_evicted_tokens
                        src_end = local_start_index + num_evicted_tokens
                        window_k_pieces.append(kv_cache["k"][:, src_start:src_end])
                        window_v_pieces.append(kv_cache["v"][:, src_start:src_end])
                    new_start = max(window_start, local_start_index) - local_start_index
                    if new_start < num_new_tokens:
                        window_k_pieces.append(roped_key[:, new_start:])
                        window_v_pieces.append(v[:, new_start:])
                    window_k = torch.cat(window_k_pieces, dim=1) if len(window_k_pieces) > 1 else window_k_pieces[0]
                    window_v = torch.cat(window_v_pieces, dim=1) if len(window_v_pieces) > 1 else window_v_pieces[0]
                    cache_update_info = {
                        "action": "roll_and_insert",
                        "sink_tokens": sink_tokens,
                        "num_rolled_tokens": num_rolled_tokens,
                        "num_evicted_tokens": num_evicted_tokens,
                        "local_start_index": local_start_index,
                        "local_end_index": local_end_index,
                        "new_k": roped_key,
                        "new_v": v,
                    }
                elif detach_cache_update:
                    with torch.no_grad():
                        kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["k"][
                            :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                        ].clone()
                        kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["v"][
                            :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                        ].clone()
                        kv_cache["k"][:, local_start_index:local_end_index] = roped_key.detach()
                        kv_cache["v"][:, local_start_index:local_end_index] = v.detach()
                    window_start = max(0, local_end_index - attention_window_size)
                    window_k_pieces = []
                    window_v_pieces = []
                    if window_start < local_start_index:
                        window_k_pieces.append(kv_cache["k"][:, window_start:local_start_index])
                        window_v_pieces.append(kv_cache["v"][:, window_start:local_start_index])
                    new_start = max(window_start, local_start_index) - local_start_index
                    if new_start < num_new_tokens:
                        window_k_pieces.append(roped_key[:, new_start:])
                        window_v_pieces.append(v[:, new_start:])
                    window_k = torch.cat(window_k_pieces, dim=1) if len(window_k_pieces) > 1 else window_k_pieces[0]
                    window_v = torch.cat(window_v_pieces, dim=1) if len(window_v_pieces) > 1 else window_v_pieces[0]
                else:
                    kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["k"][:, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["v"][:, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                    kv_cache["v"][:, local_start_index:local_end_index] = v
                    window_start = max(0, local_end_index - attention_window_size)
                    window_k = kv_cache["k"][:, window_start:local_end_index]
                    window_v = kv_cache["v"][:, window_start:local_end_index]
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                if defer_cache_update:
                    window_start = max(0, local_end_index - attention_window_size)
                    window_k_pieces = []
                    window_v_pieces = []
                    if window_start < local_start_index:
                        window_k_pieces.append(kv_cache["k"][:, window_start:local_start_index])
                        window_v_pieces.append(kv_cache["v"][:, window_start:local_start_index])
                    new_start = max(window_start, local_start_index) - local_start_index
                    if new_start < num_new_tokens:
                        window_k_pieces.append(roped_key[:, new_start:])
                        window_v_pieces.append(v[:, new_start:])
                    window_k = torch.cat(window_k_pieces, dim=1) if len(window_k_pieces) > 1 else window_k_pieces[0]
                    window_v = torch.cat(window_v_pieces, dim=1) if len(window_v_pieces) > 1 else window_v_pieces[0]
                    cache_update_info = {
                        "action": "direct_insert",
                        "local_start_index": local_start_index,
                        "local_end_index": local_end_index,
                        "new_k": roped_key,
                        "new_v": v,
                    }
                elif detach_cache_update:
                    window_start = max(0, local_end_index - attention_window_size)
                    window_k_pieces = []
                    window_v_pieces = []
                    if window_start < local_start_index:
                        window_k_pieces.append(kv_cache["k"][:, window_start:local_start_index])
                        window_v_pieces.append(kv_cache["v"][:, window_start:local_start_index])
                    new_start = max(window_start, local_start_index) - local_start_index
                    if new_start < num_new_tokens:
                        window_k_pieces.append(roped_key[:, new_start:])
                        window_v_pieces.append(v[:, new_start:])
                    window_k = torch.cat(window_k_pieces, dim=1) if len(window_k_pieces) > 1 else window_k_pieces[0]
                    window_v = torch.cat(window_v_pieces, dim=1) if len(window_v_pieces) > 1 else window_v_pieces[0]
                    with torch.no_grad():
                        kv_cache["k"][:, local_start_index:local_end_index] = roped_key.detach()
                        kv_cache["v"][:, local_start_index:local_end_index] = v.detach()
                else:
                    kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                    kv_cache["v"][:, local_start_index:local_end_index] = v
                    window_start = max(0, local_end_index - attention_window_size)
                    window_k = kv_cache["k"][:, window_start:local_end_index]
                    window_v = kv_cache["v"][:, window_start:local_end_index]
            x = attention(roped_query, window_k, window_v)
            if sequence_parallel_cache:
                x = all_to_all_4d(x, scatter_dim=1, gather_dim=2)
            if not defer_cache_update:
                kv_cache["global_end_index"].fill_(current_end)
                kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        # x.shape is [1, 65520, 1536]
        x = self.o(x)
        if kv_cache is not None and defer_cache_update:
            return x, (current_end, local_end_index, cache_update_info)
        return x


class CausalWanAttentionBlock(nn.Module):
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        local_frame_offset=None,
        balanced_sequence_parallel=False,
        defer_cache_update=False,
        detach_cache_update=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        self_attn_result = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
            local_frame_offset,
            balanced_sequence_parallel,
            defer_cache_update=defer_cache_update,
            detach_cache_update=detach_cache_update,
        )
        if isinstance(self_attn_result, tuple):
            y, cache_update_info = self_attn_result
        else:
            y = self_attn_result
            cache_update_info = None

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn((self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2))
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        if cache_update_info is not None:
            return x, cache_update_info
        return x


class CausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0])
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules = ["WanAttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        local_attn_size=-1,
        sink_size=0,
        defer_kv_cache_updates=False,
        detach_kv_cache_updates=False,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            defer_kv_cache_updates (`bool`, *optional*, defaults to False):
                Delay KV cache writes until after checkpointed blocks finish.
            detach_kv_cache_updates (`bool`, *optional*, defaults to False):
                Store detached K/V in persistent cache during grad-enabled cache updates.
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v", "ti2v", "s2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.defer_kv_cache_updates = bool(defer_kv_cache_updates)
        self.detach_kv_cache_updates = bool(detach_kv_cache_updates)
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "i2v_cross_attn" if model_type == "i2v" else "t2v_cross_attn"
        self.blocks = nn.ModuleList([CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, local_attn_size, sink_size, qk_norm, cross_attn_norm, eps) for _ in range(num_layers)])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1)

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None
        self._block_mask_batch_size = 0
        self._block_mask_key = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(
        self,
        module=None,
        value=False,
        enable=None,
        gradient_checkpointing_func=None,
    ):
        self.gradient_checkpointing = value if enable is None else enable

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(device: torch.device | str, num_frames: int = 21, frame_seqlen: int | None = None, num_frame_per_block=1, local_attn_size=-1) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        if frame_seqlen is None:
            raise ValueError("frame_seqlen must be set from the current latent resolution.")
        frame_seqlen = int(frame_seqlen)
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(start=0, end=total_length, step=frame_seqlen * num_frame_per_block, device=device)

        for tmp in frame_indices:
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length, KV_LEN=total_length + padded_length, _compile=True, device=device)

        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask_natural(
        device: torch.device | str,
        num_frames: int = 21,
        frame_seqlen: int | None = None,
        num_frame_per_block=1,
        sp_size=1,
        batch_size: int | None = None,
    ) -> BlockMask:
        if frame_seqlen is None:
            raise ValueError("frame_seqlen must be set from the current latent resolution.")
        frame_seqlen = int(frame_seqlen)
        if num_frames % sp_size != 0:
            raise ValueError(f"num_frames={num_frames} must be divisible by sp_size={sp_size} for balanced teacher forcing.")

        local_frames = num_frames // sp_size
        clean_half = local_frames * frame_seqlen
        per_rank_length = 2 * clean_half
        total_length = num_frames * frame_seqlen * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        def attention_mask(b, h, q_idx, kv_idx):
            is_real_q = q_idx < total_length
            is_real_k = kv_idx < total_length

            q_rank = q_idx // per_rank_length
            q_in_rank = q_idx % per_rank_length
            q_is_noisy = q_in_rank >= clean_half
            q_side = q_in_rank % clean_half
            q_frame = q_rank * local_frames + q_side // frame_seqlen
            q_block = q_frame // num_frame_per_block

            kv_rank = kv_idx // per_rank_length
            kv_in_rank = kv_idx % per_rank_length
            kv_is_noisy = kv_in_rank >= clean_half
            kv_side = kv_in_rank % clean_half
            kv_frame = kv_rank * local_frames + kv_side // frame_seqlen
            kv_block = kv_frame // num_frame_per_block

            clean_to_clean = (~q_is_noisy) & (~kv_is_noisy) & (kv_block <= q_block)
            noisy_to_clean = q_is_noisy & (~kv_is_noisy) & (kv_block < q_block)
            noisy_to_noisy = q_is_noisy & kv_is_noisy & (kv_block == q_block)

            eye_mask = q_idx == kv_idx
            return eye_mask | (is_real_q & is_real_k & (clean_to_clean | noisy_to_clean | noisy_to_noisy))

        return create_block_mask(
            attention_mask,
            B=batch_size,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=True,
            device=device,
        )

    @staticmethod
    def _prepare_teacher_forcing_mask(device: torch.device | str, num_frames: int = 21, frame_seqlen: int | None = None, num_frame_per_block=1) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        if frame_seqlen is None:
            raise ValueError("frame_seqlen must be set from the current latent resolution.")
        frame_seqlen = int(frame_seqlen)
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(start=0, end=num_frames * frame_seqlen, step=attention_block_size, device=device, dtype=torch.long)

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start : start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(num_frames * frame_seqlen, total_length, step=attention_block_size, device=device, dtype=torch.long)
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length, KV_LEN=total_length + padded_length, _compile=True, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2

            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255.0 * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(device: torch.device | str, num_frames: int = 21, frame_seqlen: int | None = None, num_frame_per_block=4, local_attn_size=-1) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        if frame_seqlen is None:
            raise ValueError("frame_seqlen must be set from the current latent resolution.")
        frame_seqlen = int(frame_seqlen)
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(start=frame_seqlen, end=total_length, step=frame_seqlen * num_frame_per_block, device=device)

        for idx, tmp in enumerate(frame_indices):
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length, KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _apply_cache_updates(kv_cache, cache_update_infos):
        for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
            cache = kv_cache[block_index]
            with torch.no_grad():
                if update_info is not None:
                    if update_info["action"] == "roll_and_insert":
                        sink_tokens = update_info["sink_tokens"]
                        num_rolled_tokens = update_info["num_rolled_tokens"]
                        num_evicted_tokens = update_info["num_evicted_tokens"]
                        local_start_index = update_info["local_start_index"]
                        local_end_index = update_info["local_end_index"]
                        cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = cache["k"][:, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                        cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = cache["v"][:, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                        cache["k"][:, local_start_index:local_end_index] = update_info["new_k"]
                        cache["v"][:, local_start_index:local_end_index] = update_info["new_v"]
                    elif update_info["action"] == "direct_insert":
                        local_start_index = update_info["local_start_index"]
                        local_end_index = update_info["local_end_index"]
                        cache["k"][:, local_start_index:local_end_index] = update_info["new_k"]
                        cache["v"][:, local_start_index:local_end_index] = update_info["new_v"]
                    else:
                        raise ValueError(f"Unknown KV cache update action: {update_info['action']}")

                cache["global_end_index"].fill_(current_end)
                cache["local_end_index"].fill_(local_end_index)

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        local_frame_offset=None,
        balanced_sequence_parallel=False,
        defer_cache_updates=None,
        detach_cache_updates=None,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one.

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=context_lens, block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        if defer_cache_updates is None:
            defer_cache_updates = self.defer_kv_cache_updates
        defer_cache_updates = bool(defer_cache_updates and torch.is_grad_enabled())
        if detach_cache_updates is None:
            detach_cache_updates = self.detach_kv_cache_updates
        detach_cache_updates = bool(detach_cache_updates and torch.is_grad_enabled() and not defer_cache_updates)
        if defer_cache_updates and is_sequence_parallel_enabled():
            raise ValueError("defer_kv_cache_updates does not support sequence-parallel KV cache. Set model.defer_kv_cache_updates=false or disable training.dmd.sp_cache.")
        cache_update_infos = []
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "local_frame_offset": local_frame_offset,
                        "balanced_sequence_parallel": balanced_sequence_parallel,
                        "defer_cache_update": defer_cache_updates,
                        "detach_cache_update": detach_cache_updates,
                    }
                )
                result = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "local_frame_offset": local_frame_offset,
                        "balanced_sequence_parallel": balanced_sequence_parallel,
                        "defer_cache_update": defer_cache_updates,
                        "detach_cache_update": detach_cache_updates,
                    }
                )
                result = block(x, **kwargs)

            if defer_cache_updates and isinstance(result, tuple):
                x, block_cache_update_info = result
                cache_update_infos.append((block_index, block_cache_update_info))
            else:
                x = result

        if defer_cache_updates and cache_update_infos:
            self._apply_cache_updates(kv_cache, cache_update_infos)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
        local_frame_offset=0,
        global_num_frames=None,
        balanced_sequence_parallel=False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask. Training TF passes a 5D tensor,
        # while inference passes a list of per-sample tensors.
        if isinstance(x, (list, tuple)):
            current_batch_size = len(x)
            sample_shape = x[0].shape
            local_num_frames = sample_shape[1]
            frame_seqlen = sample_shape[-2] * sample_shape[-1] // (self.patch_size[1] * self.patch_size[2])
        else:
            current_batch_size = x.shape[0]
            local_num_frames = x.shape[2]
            frame_seqlen = x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2])

        mask_num_frames = local_num_frames if global_num_frames is None else int(global_num_frames)
        mask_key = (
            current_batch_size,
            "tf" if clean_x is not None else "causal",
            mask_num_frames,
            frame_seqlen,
            self.num_frame_per_block,
            self.local_attn_size,
            self.independent_first_frame,
            bool(balanced_sequence_parallel and is_sequence_parallel_enabled()),
            get_sequence_parallel_world_size() if is_sequence_parallel_enabled() else 1,
        )
        if self.block_mask is None or self._block_mask_batch_size != current_batch_size or self._block_mask_key != mask_key:
            self._block_mask_batch_size = current_batch_size
            self._block_mask_key = mask_key
            if clean_x is not None:  # TF
                if self.independent_first_frame:
                    raise NotImplementedError()
                elif balanced_sequence_parallel and is_sequence_parallel_enabled():
                    self.block_mask = self._prepare_teacher_forcing_mask_natural(
                        device,
                        num_frames=mask_num_frames,
                        frame_seqlen=frame_seqlen,
                        num_frame_per_block=self.num_frame_per_block,
                        sp_size=get_sequence_parallel_world_size(),
                        batch_size=current_batch_size,
                    )
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device,
                        num_frames=mask_num_frames,
                        frame_seqlen=frame_seqlen,
                        num_frame_per_block=self.num_frame_per_block,
                    )
            else:  # DF?
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device,
                        num_frames=mask_num_frames,
                        frame_seqlen=frame_seqlen,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device,
                        num_frames=mask_num_frames,
                        frame_seqlen=frame_seqlen,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size,
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))], dim=1) for u in x])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            # clean_x.detach()
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            local_frame_offset=local_frame_offset,
            balanced_sequence_parallel=balanced_sequence_parallel,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)

            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2 :]
        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(self, *args, **kwargs):
        if kwargs.get("kv_cache", None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            # TF or DF
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
