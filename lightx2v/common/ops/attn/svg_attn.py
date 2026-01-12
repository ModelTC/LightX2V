import math
from functools import lru_cache
from math import ceil

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from loguru import logger
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate


@triton.jit
def wan_hidden_states_placement_kernel(
    hidden_states_ptr,  # [cfg, num_heads, seq_len, head_dim] seq_len = context_length + num_frame * frame_size
    hidden_states_out_ptr,  # [cfg, num_heads, seq_len, head_dim]
    best_mask_idx_ptr,  # [cfg, num_heads]
    hidden_states_stride_b,
    hidden_states_stride_h,
    hidden_states_stride_s,
    hidden_states_stride_d,
    mask_idx_stride_b,
    mask_idx_stride_h,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    context_length: tl.constexpr,
    num_frame: tl.constexpr,
    frame_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Copy hidden_states to output
    # range: [b, h, block_id * block_size: block_id * block_size + block_size, :]
    cfg = tl.program_id(0)
    head = tl.program_id(1)
    block_id = tl.program_id(2)

    start_id = block_id * BLOCK_SIZE
    end_id = start_id + BLOCK_SIZE
    end_id = tl.where(end_id > seq_len, seq_len, end_id)

    # Load best mask idx (0 is spatial, 1 is temporal)
    is_temporal = tl.load(best_mask_idx_ptr + cfg * mask_idx_stride_b + head * mask_idx_stride_h)

    offset_token = tl.arange(0, BLOCK_SIZE) + start_id
    offset_mask = offset_token < seq_len
    offset_d = tl.arange(0, head_dim)

    if is_temporal:
        patch_id = offset_token // num_frame
        frame_id = offset_token - patch_id * num_frame
        offset_store_token = tl.where(offset_token >= seq_len - context_length, offset_token, frame_id * frame_size + patch_id)

        offset_load = (cfg * hidden_states_stride_b + head * hidden_states_stride_h + offset_token[:, None] * hidden_states_stride_s) + offset_d[None, :] * hidden_states_stride_d
        offset_hidden_states = hidden_states_ptr + offset_load

        offset_store = (cfg * hidden_states_stride_b + head * hidden_states_stride_h + offset_store_token[:, None] * hidden_states_stride_s) + offset_d[None, :] * hidden_states_stride_d
        offset_hidden_states_out = hidden_states_out_ptr + offset_store

        # Maybe tune the pipeline here
        hidden_states = tl.load(offset_hidden_states, mask=offset_mask[:, None])
        tl.store(offset_hidden_states_out, hidden_states, mask=offset_mask[:, None])
    else:
        offset_load = (cfg * hidden_states_stride_b + head * hidden_states_stride_h + offset_token[:, None] * hidden_states_stride_s) + offset_d[None, :] * hidden_states_stride_d
        offset_hidden_states = hidden_states_ptr + offset_load

        offset_store = offset_load
        offset_hidden_states_out = hidden_states_out_ptr + offset_store

        # Maybe tune the pipeline here
        hidden_states = tl.load(offset_hidden_states, mask=offset_mask[:, None])
        tl.store(offset_hidden_states_out, hidden_states, mask=offset_mask[:, None])


def wan_hidden_states_placement(hidden_states, hidden_states_out, best_mask_idx, context_length, num_frame, frame_size):
    cfg, num_heads, seq_len, head_dim = hidden_states.shape
    BLOCK_SIZE = 128
    assert seq_len == context_length + num_frame * frame_size

    grid = (cfg, num_heads, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    wan_hidden_states_placement_kernel[grid](
        hidden_states,
        hidden_states_out,
        best_mask_idx,
        hidden_states.stride(0),
        hidden_states.stride(1),
        hidden_states.stride(2),
        hidden_states.stride(3),
        best_mask_idx.stride(0),
        best_mask_idx.stride(1),
        seq_len,
        head_dim,
        context_length,
        num_frame,
        frame_size,
        BLOCK_SIZE,
    )

    return hidden_states_out


@triton.jit
def wan_sparse_head_placement_kernel(
    query_ptr,
    key_ptr,
    value_ptr,  # [cfg, num_heads, seq_len, head_dim] seq_len = context_length + num_frame * frame_size
    query_out_ptr,
    key_out_ptr,
    value_out_ptr,  # [cfg, num_heads, seq_len, head_dim]
    best_mask_idx_ptr,  # [cfg, num_heads]
    query_stride_b,
    query_stride_h,
    query_stride_s,
    query_stride_d,
    mask_idx_stride_b,
    mask_idx_stride_h,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    context_length: tl.constexpr,
    num_frame: tl.constexpr,
    frame_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Copy query, key, value to output
    # range: [b, h, block_id * block_size: block_id * block_size + block_size, :]
    cfg = tl.program_id(0)
    head = tl.program_id(1)
    block_id = tl.program_id(2)

    start_id = block_id * BLOCK_SIZE
    end_id = start_id + BLOCK_SIZE
    end_id = tl.where(end_id > seq_len, seq_len, end_id)

    # Load best mask idx (0 is spatial, 1 is temporal)
    is_temporal = tl.load(best_mask_idx_ptr + cfg * mask_idx_stride_b + head * mask_idx_stride_h)

    offset_token = tl.arange(0, BLOCK_SIZE) + start_id
    offset_mask = offset_token < seq_len
    offset_d = tl.arange(0, head_dim)

    if is_temporal:
        frame_id = offset_token // frame_size
        patch_id = offset_token - frame_id * frame_size
        offset_store_token = tl.where(offset_token >= seq_len - context_length, offset_token, patch_id * num_frame + frame_id)

        offset_load = (cfg * query_stride_b + head * query_stride_h + offset_token[:, None] * query_stride_s) + offset_d[None, :] * query_stride_d
        offset_query = query_ptr + offset_load
        offset_key = key_ptr + offset_load
        offset_value = value_ptr + offset_load

        offset_store = (cfg * query_stride_b + head * query_stride_h + offset_store_token[:, None] * query_stride_s) + offset_d[None, :] * query_stride_d
        offset_query_out = query_out_ptr + offset_store
        offset_key_out = key_out_ptr + offset_store
        offset_value_out = value_out_ptr + offset_store

        # Maybe tune the pipeline here
        query = tl.load(offset_query, mask=offset_mask[:, None])
        tl.store(offset_query_out, query, mask=offset_mask[:, None])
        key = tl.load(offset_key, mask=offset_mask[:, None])
        tl.store(offset_key_out, key, mask=offset_mask[:, None])
        value = tl.load(offset_value, mask=offset_mask[:, None])
        tl.store(offset_value_out, value, mask=offset_mask[:, None])

    else:
        offset_load = (cfg * query_stride_b + head * query_stride_h + offset_token[:, None] * query_stride_s) + offset_d[None, :] * query_stride_d
        offset_query = query_ptr + offset_load
        offset_key = key_ptr + offset_load
        offset_value = value_ptr + offset_load

        offset_store = offset_load
        offset_query_out = query_out_ptr + offset_store
        offset_key_out = key_out_ptr + offset_store
        offset_value_out = value_out_ptr + offset_store

        # Maybe tune the pipeline here
        query = tl.load(offset_query, mask=offset_mask[:, None])
        tl.store(offset_query_out, query, mask=offset_mask[:, None])
        key = tl.load(offset_key, mask=offset_mask[:, None])
        tl.store(offset_key_out, key, mask=offset_mask[:, None])
        value = tl.load(offset_value, mask=offset_mask[:, None])
        tl.store(offset_value_out, value, mask=offset_mask[:, None])


def wan_sparse_head_placement(query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size):
    cfg, num_heads, seq_len, head_dim = query.shape
    BLOCK_SIZE = 128
    assert seq_len == context_length + num_frame * frame_size

    grid = (cfg, num_heads, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    wan_sparse_head_placement_kernel[grid](
        query,
        key,
        value,
        query_out,
        key_out,
        value_out,
        best_mask_idx,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        best_mask_idx.stride(0),
        best_mask_idx.stride(1),
        seq_len,
        head_dim,
        context_length,
        num_frame,
        frame_size,
        BLOCK_SIZE,
    )


def generate_temporal_head_mask_mod(context_length: int = 226, prompt_length: int = 226, num_frames: int = 13, token_per_frame: int = 1350, mul: int = 2):
    def round_to_multiple(idx):
        return ceil(idx / 128) * 128

    def temporal_mask_mod(b, h, q_idx, kv_idx):
        two_frame = round_to_multiple(mul * token_per_frame)
        temporal_head_mask = torch.abs(q_idx - kv_idx) <= two_frame

        # return temporal_head_mask
        first_frame_mask = kv_idx < token_per_frame
        video_mask = first_frame_mask | temporal_head_mask
        return video_mask

    return temporal_mask_mod


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda", _compile=False):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device, _compile=_compile)
    return block_mask


def prepare_flexattention(cfg_size, num_head, head_dim, dtype, device, context_length, prompt_length, num_frame, frame_size, diag_width=1, multiplier=2):
    assert diag_width == multiplier, f"{diag_width} is not equivalent to {multiplier}"
    seq_len = context_length + num_frame * frame_size
    mask_mod = generate_temporal_head_mask_mod(context_length, prompt_length, num_frame, frame_size, mul=multiplier)
    block_mask = create_block_mask_cached(mask_mod, None, None, seq_len, seq_len, device=device, _compile=True)
    return block_mask


def sparsity_to_width(sparsity, context_length, num_frame, frame_size):
    seq_len = context_length + num_frame * frame_size
    total_elements = seq_len**2

    sparsity = (sparsity * total_elements - 2 * seq_len * context_length) / total_elements

    width = seq_len * (1 - math.sqrt(1 - sparsity))
    width_frame = width / frame_size

    return width_frame


def get_attention_mask(mask_name, sample_mse_max_row, context_length, num_frame, frame_size):
    attention_mask = torch.zeros((context_length + num_frame * frame_size, context_length + num_frame * frame_size), device="cpu")

    # TODO: fix hard coded mask
    if mask_name == "spatial":
        pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.bool, device="cpu")

        pixel_attn_mask[:, :frame_size] = 1  # First Frame Sink

        block_size, block_thres = 128, frame_size * 2
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1
        attention_mask = pixel_attn_mask
    else:
        pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.bool, device="cpu")

        pixel_attn_mask[:, :frame_size] = 1  # First Frame Sink

        block_size, block_thres = 128, frame_size * 2
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1

        pixel_attn_mask = pixel_attn_mask.reshape(frame_size, num_frame, frame_size, num_frame).permute(1, 0, 3, 2).reshape(frame_size * num_frame, frame_size * num_frame)
        attention_mask = pixel_attn_mask

    attention_mask = attention_mask[:sample_mse_max_row].cuda()
    return attention_mask


@ATTN_WEIGHT_REGISTER("svg_attn")
class SvgAttnWeight(AttnWeightTemplate):
    head_num = None
    head_dim = None
    sample_mse_max_row = None
    num_sampled_rows = None
    context_length = None
    attnmap_frame_num = None
    seqlen = None
    sparsity = None
    mask_name_list = ["spatial", "temporal"]
    attention_masks = None
    block_mask = None

    @classmethod
    def prepare(cls, head_num, head_dim, sample_mse_max_row, num_sampled_rows, context_length, sparsity):
        cls.head_num = head_num
        cls.head_dim = head_dim
        cls.sample_mse_max_row = sample_mse_max_row
        cls.num_sampled_rows = num_sampled_rows
        cls.context_length = context_length
        cls.sparsity = sparsity
        torch._dynamo.config.cache_size_limit = 192 * 3
        torch._dynamo.config.accumulated_cache_size_limit = 192 * 3
        logger.info(
            f"SvgAttnWeight Prepare: head_num={head_num}, head_dim={head_dim}, sample_mse_max_row={sample_mse_max_row}, num_sampled_rows={num_sampled_rows}, context_length={context_length}, sparsity={sparsity}"
        )

    def __init__(self):
        self.config = {}
        self.sparse_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

    @classmethod
    def prepare_mask(cls, seqlen):
        # Use class attributes so updates affect all instances of this class
        if seqlen == cls.seqlen:
            return
        frame_size = seqlen // cls.attnmap_frame_num
        cls.attention_masks = [get_attention_mask(mask_name, cls.sample_mse_max_row, cls.context_length, cls.attnmap_frame_num, frame_size) for mask_name in cls.mask_name_list]
        multiplier = diag_width = sparsity_to_width(cls.sparsity, cls.context_length, cls.attnmap_frame_num, frame_size)
        cls.block_mask = prepare_flexattention(
            1, cls.head_num, cls.head_dim, torch.bfloat16, "cuda", cls.context_length, cls.context_length, cls.attnmap_frame_num, frame_size, diag_width=diag_width, multiplier=multiplier
        )
        cls.seqlen = seqlen
        logger.info(f"SvgAttnWeight Update: seqlen={seqlen}")

    def apply(
        self,
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        **kwargs,
    ):
        q = q.unsqueeze(0).transpose(1, 2)
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)
        bs, num_heads, seq_len, dim = q.size()

        self.prepare_mask(seq_len)
        sampled_mses = self.sample_mse(q, k, v)
        best_mask_idx = torch.argmin(sampled_mses, dim=0)

        output_hidden_states = torch.zeros_like(q)
        query_out, key_out, value_out = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)

        query_out, key_out, value_out = self.fast_sparse_head_placement(
            q, k, v, query_out, key_out, value_out, best_mask_idx, self.context_length, self.attnmap_frame_num, seq_len // self.attnmap_frame_num
        )

        hidden_states = self.sparse_attention(query_out, key_out, value_out)
        wan_hidden_states_placement(hidden_states, output_hidden_states, best_mask_idx, self.context_length, self.attnmap_frame_num, seq_len // self.attnmap_frame_num)

        return output_hidden_states.reshape(bs, num_heads, seq_len, dim).transpose(1, 2).reshape(bs * seq_len, -1)

    def fast_sparse_head_placement(self, query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size):
        wan_sparse_head_placement(query, key, value, query_out, key_out, value_out, best_mask_idx, context_length, num_frame, frame_size)
        return query_out, key_out, value_out

    def sample_mse(self, query, key, value):
        cfg, num_heads, seq_len, dim = query.size()
        num_sampled_rows = min(self.num_sampled_rows, seq_len)
        sampled_rows = torch.randint(low=0, high=self.sample_mse_max_row, size=(num_sampled_rows,))
        sampled_q = query[:, :, sampled_rows, :]
        sampled_qk_scores = torch.matmul(sampled_q, key.transpose(-2, -1)) / (dim**0.5)

        sampled_attn_weights = F.softmax(sampled_qk_scores, dim=-1)
        sampled_golden_hidden_states = torch.matmul(sampled_attn_weights, value)  # (1, seq_len, dim)

        sampled_mses = torch.zeros(len(self.attention_masks), cfg, num_heads, device=query.device, dtype=query.dtype)

        # Only have Tri-diagonal and Striped
        for mask_idx, attn_mask in enumerate(self.attention_masks):
            sampled_attention_mask = attn_mask[sampled_rows, :]
            sampled_attention_scores = sampled_qk_scores.masked_fill(sampled_attention_mask == 0, float("-inf"))
            sampled_attn_weights = F.softmax(sampled_attention_scores, dim=-1)
            sampled_hidden_states = torch.matmul(sampled_attn_weights, value)
            mse = torch.mean((sampled_hidden_states - sampled_golden_hidden_states) ** 2, dim=(2, 3))
            sampled_mses[mask_idx] = mse

        return sampled_mses


if __name__ == "__main__":
    q, k, v = torch.randn(32130, 40, 128, dtype=torch.bfloat16).cuda(), torch.randn(32130, 40, 128, dtype=torch.bfloat16).cuda(), torch.randn(32130, 40, 128, dtype=torch.bfloat16).cuda()

    SvgAttnWeight.prepare(head_num=40, head_dim=128, sample_mse_max_row=10000, num_sampled_rows=64, context_length=0, sparsity=0.25)
    svg_attn = SvgAttnWeight()
    print("SvgAttnWeight initialized.")

    out = svg_attn.apply(q, k, v)
    print(f"out: {out.shape}, {out.dtype}, {out.device}")
