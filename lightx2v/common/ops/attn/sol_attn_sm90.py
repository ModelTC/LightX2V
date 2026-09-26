"""SM90 Sol-Attn query ranges with unchanged 64-token routing coordinates.

The KV-tail recipe follows NVlabs/Sana sol-engine 71350faed59a (Apache-2.0).
The upstream mainloop is inherited without changing its attention arithmetic.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from sol_attn._vendor.flash_attn.cute.seqlen_info import SeqlenInfoQK
from sol_attn.common import to_cute_tensor
from sol_attn.interface import _sink_block_range, _stream, _validate_inputs
from sol_attn.preprocess import prepare
from sol_attn.sm90.mainloop import SolAttnMainloopSm90


class SolAttnQueryRangeSm90(SolAttnMainloopSm90):
    def __init__(self, tokens, query_block_start):
        # Keep the released KV-tail recipe; short Q does not change KV blocks.
        blocks = (tokens + 63) // 64
        full_groups, tail = divmod(blocks, 64)
        has_full_groups = tail == 0
        has_full_blocks = tokens % 64 == 0
        super().__init__(
            cutlass.BFloat16,
            head_dim=128,
            head_dim_v=128,
            qhead_per_kvhead=1,
            is_causal=False,
            is_local=False,
            pack_gqa=False,
            tile_m=64,
            tile_n=64,
            num_stages=1,
            num_threads=128,
            sol_attn_assume_lane_group_route_reduce=has_full_blocks and has_full_groups,
            sol_attn_assume_full_k_exact_blocks=has_full_blocks,
            sol_attn_tail_exact_words1=0 < tail <= 8,
            sol_attn_assume_full_route_groups=has_full_groups,
            sol_attn_static_num_full_route_groups=-1 if has_full_groups else full_groups,
            sol_attn_static_tail_valid_count=-1 if has_full_groups else tail,
            sol_attn_tail_physical_tile16=0 < tail <= 16,
            sol_attn_exact_mask_seqlen_last_only=not has_full_blocks,
            sol_attn_tail16_lane_group_route_reduce=tail == 16,
            sol_attn_num_splits=1,
        )
        self.query_block_start = query_block_start

    @cute.jit
    def sol_attn_build_route_mask_from_acc(
        self,
        acc_S: cute.Tensor,
        route_sums: cute.Tensor,
        tScS_mn: cute.Tensor,
        m_block: Int32,
        group_start_n_block: Int32,
        valid_count: Int32,
        route_col_offset: Int32,
        seqlen: SeqlenInfoQK,
        batch_idx: Int32,
        head_idx: Int32,
        mGlobalThresh: cute.Tensor,
        softmax_scale_log2: Float32,
        sink_range: Int32,
        assume_full_route_group: cutlass.Constexpr[bool] = False,
        physical_route_tile_n: cutlass.Constexpr[int] = 64,
        route_mask_words_override: cutlass.Constexpr[int] = 0,
    ):
        # Only routing uses absolute query indices. Q loads and O stores keep
        # their local coordinates. Shift the limit with the index so row masks
        # and q_len stay identical, including a partial final query block.
        absolute_seqlen = SeqlenInfoQK(
            seqlen.offset_q,
            seqlen.offset_k,
            seqlen.padded_offset_q,
            seqlen.padded_offset_k,
            seqlen.seqlen_q + self.query_block_start * self.tile_m,
            seqlen.seqlen_k,
            seqlen.has_cu_seqlens_q,
            seqlen.has_cu_seqlens_k,
            seqlen.has_seqused_q,
            seqlen.has_seqused_k,
        )
        return SolAttnMainloopSm90.sol_attn_build_route_mask_from_acc(
            self,
            acc_S,
            route_sums,
            tScS_mn,
            m_block + self.query_block_start,
            group_start_n_block,
            valid_count,
            route_col_offset,
            absolute_seqlen,
            batch_idx,
            head_idx,
            mGlobalThresh,
            softmax_scale_log2,
            sink_range,
            assume_full_route_group=assume_full_route_group,
            physical_route_tile_n=physical_route_tile_n,
            route_mask_words_override=route_mask_words_override,
        )


_COMPILED = {}


def sol_attn_query_range(q, k, v, *, query_range, scale, tau, thresh_type, sink_tokens, sink_start):
    """Compute only the query blocks intersecting the requested token range."""
    _validate_inputs(q, k, v, thresh_type, sink_tokens, sink_start)
    batch, tokens, heads, _ = q.shape
    query_start, query_end = query_range
    first_block = query_start // 64
    aligned_start = first_block * 64
    aligned_end = min((query_end + 63) // 64 * 64, tokens)
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    with torch.cuda.device(q.device):
        # Full-Q pooling preserves the original routing thresholds exactly.
        kc, vc, threshold = prepare(q, k, v, scale=scale, tau=float(tau), thresh_type=thresh_type)
        short_q = q[:, aligned_start:aligned_end].contiguous()
        output = torch.empty_like(short_q)
        lse = torch.empty((batch, short_q.shape[1], heads), device=q.device, dtype=torch.float32)
        tensors = [short_q, k, v, output, kc, vc, threshold, lse]
        args = [to_cute_tensor(tensor) for tensor in tensors]
        stream = _stream(q.device)
        sink_range = 0
        if sink_tokens:
            sink_lo, sink_hi = _sink_block_range(tokens, sink_start, sink_tokens)
            sink_range = sink_lo | (sink_hi << 16)
        key = (q.device.index, batch, tokens, heads, aligned_start, aligned_end)
        compiled = _COMPILED.get(key)
        if compiled is None:
            operator = SolAttnQueryRangeSm90(tokens, first_block)
            compiled = cute.compile(operator, *args, scale, sink_range, stream=stream, options="--enable-tvm-ffi")
            _COMPILED[key] = compiled
        compiled(*args, scale, sink_range, stream=stream)
    return output[:, query_start - aligned_start : query_end - aligned_start]
