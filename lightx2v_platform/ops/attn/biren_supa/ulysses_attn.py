import os

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v_platform.ops.attn.template import AttnWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_ATTN_WEIGHT_REGISTER

try:
    import torch_br
    from torch_br.utils.tensor_methods import Sbp
except ImportError:
    torch_br = None
    Sbp = None

from lightx2v_platform_biren import _lightx2v_kernel_biren as _wan_native

if _wan_native is not None and not all(
    hasattr(_wan_native, name) for name in ("wan_chunk_to_buffers", "wan_buffers_to_cat")
):
    # An older installed extension may not provide the Wan native operations.
    # Treat it like an unavailable extension until `build.sh` is rerun.
    _wan_native = None

_WAN_NATIVE_USP_ENABLED = os.environ.get("LIGHTX2V_WAN_NATIVE_USP", "1") != "0"


@PLATFORM_ATTN_WEIGHT_REGISTER("biren_ulysses")
class BirenUlyssesAttnWeight(AttnWeightTemplate):
    """
    Biren-specific Ulysses sequence parallel attention.
    Matches Wan2.2's distributed_attention + sp_attn_forward pattern.
    Uses supa_shape_transform_qkv for SBP-aware reshaping when in TP-2die mode.
    """

    def __init__(self):
        if torch_br is None:
            raise ImportError("torch_br is not installed.")
        self.config = {}

    def apply(
        self,
        q,
        k,
        v,
        slice_qkv_len,
        cu_seqlens_qkv,
        attention_module=None,
        seq_p_group=None,
        use_fp8_comm=False,
        use_fp4_comm=False,
        use_tensor_fusion=False,
        enable_head_parallel=False,
        img_first=True,
        q_only_img=False,
        **kwargs,
    ):
        world_size = dist.get_world_size(seq_p_group)

        # TP-2die SBP mode: q/k/v come in as [1, s, n*d] with SBP format
        if len(q.shape) == 3 and q.shape[0] == 1:
            is_sbp = hasattr(q, 'sbp') and callable(q.sbp) and not q.sbp().is_bb
            if not is_sbp and hasattr(v, 'sbp') and callable(v.sbp) and not v.sbp().is_bb:
                is_sbp = True
            if is_sbp:
                num_heads = kwargs.get('num_heads', 40)
                head_dim = kwargs.get('head_dim', q.shape[2] // num_heads)
                return self._apply_sbp_mode(q, k, v, slice_qkv_len, cu_seqlens_qkv,
                                            attention_module, seq_p_group, world_size,
                                            num_heads, head_dim, img_first)

        if len(q.shape) == 4:
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])

        num_heads = q.shape[1]
        head_dim = q.shape[2]

        return self._apply_bb_mode(q, k, v, slice_qkv_len, cu_seqlens_qkv,
                                   attention_module, seq_p_group, world_size,
                                   num_heads, head_dim, img_first)

    def _apply_sbp_mode(self, q, k, v, slice_qkv_len, cu_seqlens_qkv,
                        attention_module, seq_p_group, world_size, num_heads, head_dim, img_first):
        """TP-2die mode: use supa_shape_transform_qkv for SBP-aware reshape.

        Input: q/k [1, s, n*d] BB format, v [1, s, n*d] S2B format
        Reference flow:
          [1, s, n*d] → supa_shape_transform_qkv(forward) → [n, s, d] S0B
          → all-to-all → attention → all-to-all
          → supa_shape_transform_qkv(inverse) → [1, s, n*d] S2B
        """
        b = q.shape[0]  # 1
        shard_seq = q.shape[1]

        # shape_transpose_pre_attn: [b, s, n*d] -> [b*n, s, d] with BB/S2B -> S0B
        q_s0b = torch_br.supa_shape_transform_qkv(q, b, shard_seq, num_heads, head_dim, True)
        k_s0b = torch_br.supa_shape_transform_qkv(k, b, shard_seq, num_heads, head_dim, True)
        v_s0b = torch_br.supa_shape_transform_qkv(v, b, shard_seq, num_heads, head_dim, True)

        # Distributed attention (all-to-all on S0B tensors)
        x = self._distributed_attention_sbp(q_s0b, k_s0b, v_s0b, shard_seq,
                                            attention_module, seq_p_group, world_size,
                                            num_heads, head_dim)

        # shape_transpose_post_attn: [b*n, s, d] S0B -> [b, s, n*d] S2B
        x_s2b = torch_br.supa_shape_transform_qkv(x, b, shard_seq, num_heads, head_dim, False, False, Sbp.sb(2))

        # Squeeze batch dim: [1, s, n*d] -> [s, n*d]
        return x_s2b.squeeze(0)

    def _distributed_attention_sbp(self, q, k, v, shard_seq, attention_module,
                                   seq_p_group, world_size, num_heads, head_dim):
        """Ulysses all-to-all on S0B tensors: scatter heads, gather sequence."""
        # All-to-all: scatter dim=0 (heads), gather dim=1 (sequence)
        q_gathered = self._all_to_all(q, scatter_dim=0, gather_dim=1, group=seq_p_group)
        k_gathered = self._all_to_all(k, scatter_dim=0, gather_dim=1, group=seq_p_group)
        v_gathered = self._all_to_all(v, scatter_dim=0, gather_dim=1, group=seq_p_group)

        # After all-to-all: [n/sp, s*sp, d]

        # Run attention on [heads, seq, dim], matching Wan2.2's BR path.
        # The tensors are already BF16 in the normal Wan path.  Keep the
        # conversion conditional so every block does not enqueue a redundant
        # device copy before FMHA.
        if q_gathered.dtype != torch.bfloat16:
            q_gathered = q_gathered.to(torch.bfloat16)
        if k_gathered.dtype != torch.bfloat16:
            k_gathered = k_gathered.to(torch.bfloat16)
        if v_gathered.dtype != torch.bfloat16:
            v_gathered = v_gathered.to(torch.bfloat16)
        attn_out, _ = torch_br.sueager_scaled_dot_product_attention_fwd(
            q_gathered,
            k_gathered,
            v_gathered,
            mask=None,
            dropout_prob=0,
            is_causal=False,
            scale=None,
            algorithm='FMHA',
        )

        # Reverse all-to-all: scatter sequence, gather heads
        result = self._all_to_all(attn_out, scatter_dim=1, gather_dim=0, group=seq_p_group)
        # result: [n, s, d] S0B
        return result

    def _all_to_all(self, input_t, scatter_dim, gather_dim, group):
        """All-to-all matching reference Wan2.2 implementation for SBP tensors."""
        world_size = dist.get_world_size(group)

        if hasattr(input_t, 'sbp') and callable(input_t.sbp) and input_t.sbp().is_bb:
            # BB path
            inputs = [u.contiguous() for u in input_t.chunk(world_size, dim=scatter_dim)]
            inputs = [torch_br._empty_with_sbp(u.shape, dtype=u.dtype, tensor_type="buffer_any", device=u.device).copy_(u) for u in inputs]
            outputs = [torch.empty_like(u) for u in inputs]
            dist.all_to_all(outputs, inputs, group=group)
            return torch.cat(outputs, dim=gather_dim).contiguous()
        else:
            # SBP path (S0B etc.) - convert to buffer_any for communication
            transfer_shape = [s if i != scatter_dim else s // world_size for i, s in enumerate(input_t.shape)]

            # The Python chunk/clone/view path materializes a reorder for every
            # USP exchange.  The native Wan kernels write/read the flat
            # BUFFER_ANY tensors directly in COLMAJOR layout, matching the
            # diffusers-br collective path.  Keep the fallback for unsupported
            # shapes, architectures, or Python-only environments.
            cat_output_shape = list(input_t.shape)
            cat_output_shape[gather_dim] = transfer_shape[gather_dim] * world_size
            cat_output_shape[scatter_dim] = transfer_shape[scatter_dim]
            if (
                _WAN_NATIVE_USP_ENABLED
                and _wan_native is not None
                and world_size in (2, 4)
                and cat_output_shape[0] % 2 == 0
            ):
                input_bufs = []
                for _ in range(world_size):
                    buf = torch_br._empty_with_sbp(
                        transfer_shape,
                        dtype=input_t.dtype,
                        axis=0,
                        sbp="sb",
                        tensor_type="colmajor",
                        device=input_t.device,
                    )
                    input_bufs.append(buf.view_as_buffer())
                _wan_native.wan_chunk_to_buffers(input_t, input_bufs, world_size, scatter_dim)
                output_bufs = [torch.empty_like(buf) for buf in input_bufs]
                dist.all_to_all(output_bufs, input_bufs, group=group)
                cat_output = torch_br._empty_with_sbp(
                    cat_output_shape,
                    dtype=input_t.dtype,
                    axis=0,
                    sbp="sb",
                    tensor_type="colmajor",
                    device=input_t.device,
                )
                _wan_native.wan_buffers_to_cat(output_bufs, cat_output, world_size, gather_dim)
                return cat_output

            inputs = [u.contiguous().clone().view_as_buffer() for u in input_t.chunk(world_size, dim=scatter_dim)]
            outputs = [torch.empty_like(u) for u in inputs]
            dist.all_to_all(outputs, inputs, group=group)
            outputs = [u.view_as_colmajor(transfer_shape, Sbp.sb(0)) for u in outputs]
            return torch.cat(outputs, dim=gather_dim).contiguous()

    def _apply_bb_mode(self, q, k, v, slice_qkv_len, cu_seqlens_qkv,
                       attention_module, seq_p_group, world_size, num_heads, head_dim, img_first):
        """Standard BB mode (non-TP-2die): original implementation."""
        if img_first:
            img_qkv_len = slice_qkv_len
            if len(cu_seqlens_qkv) == 3:
                txt_qkv_len = cu_seqlens_qkv[1] - slice_qkv_len
            elif len(cu_seqlens_qkv) == 2:
                txt_qkv_len = cu_seqlens_qkv[1] - slice_qkv_len
        else:
            txt_qkv_len = slice_qkv_len
            img_qkv_len = cu_seqlens_qkv[1] - slice_qkv_len

        if img_first:
            img_q = q[:img_qkv_len].contiguous()
            img_k = k[:img_qkv_len].contiguous()
            img_v = v[:img_qkv_len].contiguous()
            txt_q = q[img_qkv_len:].contiguous()
            txt_k = k[img_qkv_len:].contiguous()
            txt_v = v[img_qkv_len:].contiguous()
        else:
            txt_q = q[:txt_qkv_len].contiguous()
            txt_k = k[:txt_qkv_len].contiguous()
            txt_v = v[:txt_qkv_len].contiguous()
            img_q = q[txt_qkv_len:].contiguous()
            img_k = k[txt_qkv_len:].contiguous()
            img_v = v[txt_qkv_len:].contiguous()

        global_img_seqlen = img_qkv_len * world_size
        shard_heads = num_heads // world_size

        img_q_r = img_q.reshape(img_qkv_len, world_size, shard_heads, head_dim)
        img_k_r = img_k.reshape(img_qkv_len, world_size, shard_heads, head_dim)
        img_v_r = img_v.reshape(img_qkv_len, world_size, shard_heads, head_dim)

        img_q_p = img_q_r.permute(1, 0, 2, 3).contiguous()
        img_k_p = img_k_r.permute(1, 0, 2, 3).contiguous()
        img_v_p = img_v_r.permute(1, 0, 2, 3).contiguous()

        def _br_all_to_all(input_t, group, world_size):
            input_list = list(input_t.chunk(world_size, dim=0))
            output_list = [torch.empty_like(x) for x in input_list]
            dist.all_to_all(output_list, input_list, group=group)
            return torch.cat(output_list, dim=0)

        output_q = _br_all_to_all(img_q_p, seq_p_group, world_size)
        output_k = _br_all_to_all(img_k_p, seq_p_group, world_size)
        output_v = _br_all_to_all(img_v_p, seq_p_group, world_size)

        gathered_q = output_q.permute(1, 0, 2, 3).reshape(global_img_seqlen, shard_heads, head_dim)
        gathered_k = output_k.permute(1, 0, 2, 3).reshape(global_img_seqlen, shard_heads, head_dim)
        gathered_v = output_v.permute(1, 0, 2, 3).reshape(global_img_seqlen, shard_heads, head_dim)

        if txt_qkv_len > 0:
            txt_q_shard = txt_q[:, :shard_heads, :].contiguous()
            txt_k_shard = txt_k[:, :shard_heads, :].contiguous()
            txt_v_shard = txt_v[:, :shard_heads, :].contiguous()
            full_q = torch.cat([gathered_q, txt_q_shard], dim=0)
            full_k = torch.cat([gathered_k, txt_k_shard], dim=0)
            full_v = torch.cat([gathered_v, txt_v_shard], dim=0)
            full_seq_len = global_img_seqlen + txt_qkv_len
        else:
            full_q = gathered_q
            full_k = gathered_k
            full_v = gathered_v
            full_seq_len = global_img_seqlen

        cu_seqlens = torch.tensor([0, full_seq_len], dtype=torch.int32, device=q.device)
        attn_out = attention_module.apply(
            q=full_q,
            k=full_k,
            v=full_v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=full_seq_len,
            max_seqlen_kv=full_seq_len,
        )

        img_attn_out = attn_out[:global_img_seqlen].reshape(global_img_seqlen, shard_heads, head_dim)
        img_attn_r = img_attn_out.reshape(world_size, img_qkv_len, shard_heads, head_dim).contiguous()
        output_attn = _br_all_to_all(img_attn_r, seq_p_group, world_size)
        img_result = output_attn.permute(1, 0, 2, 3).reshape(img_qkv_len, num_heads, head_dim)

        if txt_qkv_len > 0:
            txt_attn_out = attn_out[global_img_seqlen:]
            txt_result_shard = txt_attn_out.reshape(txt_qkv_len, shard_heads, head_dim)
            txt_result_full = torch.zeros(txt_qkv_len, num_heads, head_dim, dtype=txt_result_shard.dtype, device=txt_result_shard.device)
            cur_rank = dist.get_rank(seq_p_group)
            txt_result_full[:, cur_rank * shard_heads:(cur_rank + 1) * shard_heads, :] = txt_result_shard
            dist.all_reduce(txt_result_full, group=seq_p_group)

            if img_first:
                result = torch.cat([img_result, txt_result_full], dim=0)
            else:
                result = torch.cat([txt_result_full, img_result], dim=0)
        else:
            result = img_result

        return result.reshape(result.shape[0], num_heads * head_dim)
