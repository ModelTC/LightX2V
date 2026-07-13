from dataclasses import dataclass

import torch
import torch.distributed as dist

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .kernels.opt_ulysses_layout import (
    attn_post,
    attn_post_fp8,
    attn_pre,
    attn_pre_fp8,
    qkv_post,
    qkv_post_fp8,
    qkv_post_head,
    qkv_post_head_fp8,
    qkv_pre,
    qkv_pre_fp8,
    qkv_pre_heads,
    qkv_pre_heads_fp8,
    qonly_qkv_post,
    qonly_qkv_post_fp8,
    qonly_qkv_post_head,
    qonly_qkv_post_head_fp8,
    qonly_qkv_pre,
    qonly_qkv_pre_fp8,
    qonly_qkv_pre_heads,
    qonly_qkv_pre_heads_fp8,
)
from .template import AttnWeightTemplate


@dataclass(frozen=True)
class _UlyssesContext:
    world_size: int
    cur_rank: int
    img_len: int
    txt_len: int
    txt_mask_len: int | None
    shard_heads: int
    hidden_dims: int
    global_img_len: int
    img_start: int
    txt_start: int
    img_first: bool
    q_only_img: bool
    use_fp8_comm: bool
    seq_p_group: object
    cu_seqlens_q: torch.Tensor
    cu_seqlens_kv: torch.Tensor
    max_seqlen_q: int
    max_seqlen_kv: int

    @property
    def txt_storage_len(self):
        return self.txt_mask_len if self.txt_mask_len is not None else self.txt_len


@ATTN_WEIGHT_REGISTER("ulysses-opt")
class OptUlyssesAttnWeight(AttnWeightTemplate):
    """Optimized Ulysses attention with fused pre/post communication stages."""

    def __init__(self):
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
        if len(q.shape) == 4:
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])

        world_size = dist.get_world_size(seq_p_group)
        cur_rank = dist.get_rank(seq_p_group)
        img_len, txt_len, txt_mask_len = self._parse_lengths(slice_qkv_len, cu_seqlens_qkv, img_first)
        _, q_heads, hidden_dims = q.shape
        _, kv_heads, _ = k.shape

        unsupported = []
        if use_fp4_comm:
            unsupported.append("use_fp4_comm=True")
        if not use_tensor_fusion:
            unsupported.append("use_tensor_fusion=False")
        if q_heads != kv_heads:
            unsupported.append(f"q_heads={q_heads} != kv_heads={kv_heads}")
        if unsupported:
            hints = []
            if not use_tensor_fusion:
                hints.append("ulysses-opt requires tensor fusion; set parallel.seq_p_tensor_fusion=true (use_tensor_fusion=True) to enable this path.")
            hints.append("Set parallel.seq_p_attn_type='ulysses' to use the legacy Ulysses implementation for unsupported cases.")
            raise NotImplementedError("ulysses-opt does not support " + ", ".join(unsupported) + ". " + " ".join(hints))

        if q_heads % world_size != 0:
            raise ValueError(f"q_heads={q_heads} must be divisible by world_size={world_size}.")

        ctx = self._build_context(
            world_size=world_size,
            cur_rank=cur_rank,
            img_len=img_len,
            txt_len=txt_len,
            txt_mask_len=txt_mask_len,
            q_heads=q_heads,
            hidden_dims=hidden_dims,
            img_first=img_first,
            q_only_img=q_only_img,
            use_fp8_comm=use_fp8_comm,
            seq_p_group=seq_p_group,
        )

        if enable_head_parallel:
            return self._apply_head_parallel(q, k, v, ctx, attention_module, kwargs)
        return self._apply_default(q, k, v, ctx, attention_module, kwargs)

    @staticmethod
    def _apply_default(q, k, v, ctx, attention_module, attention_kwargs):
        if ctx.q_only_img:
            q, k, v = OptUlyssesAttnWeight._qonly_qkv_a2a(q, k, v, ctx)
        else:
            q, k, v = OptUlyssesAttnWeight._qkv_a2a(q, k, v, ctx)

        attn = attention_module.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=ctx.cu_seqlens_q,
            cu_seqlens_kv=ctx.cu_seqlens_kv,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_kv=ctx.max_seqlen_kv,
            **attention_kwargs,
        )

        if ctx.q_only_img:
            return OptUlyssesAttnWeight._img_attn_a2a(attn, ctx)
        return OptUlyssesAttnWeight._attn_a2a(attn, ctx)

    @staticmethod
    def _apply_head_parallel(q, k, v, ctx, attention_module, attention_kwargs):
        """Run Ulysses with one independent communication stream per local head."""
        # Keep the legacy head-parallel ordering: enqueue every local-head qkv
        # all-to-all first, then wait one head at a time before qkv_post and
        # attention. This preserves the old comm/compute overlap pattern.
        qkv_records = []
        # Read all per-rank head lanes directly from the original full-head
        # tensors in one pre-kernel launch. Each local-head slab remains
        # contiguous and is still communicated independently below.
        if ctx.use_fp8_comm:
            if ctx.q_only_img:
                qkv_payload_heads, qkv_scale_heads, qkv_shape, qkv_scale_shape = qonly_qkv_pre_heads_fp8(q, k, v, ctx.img_start, ctx.img_len, ctx.world_size, ctx.shard_heads)
            else:
                qkv_payload_heads, qkv_scale_heads, qkv_shape, qkv_scale_shape = qkv_pre_heads_fp8(q, k, v, ctx.img_start, ctx.img_len, ctx.world_size, ctx.shard_heads)
            for local_head in range(ctx.shard_heads):
                qkv_payload = qkv_payload_heads[local_head]
                qkv_scale = qkv_scale_heads[local_head]
                output_qkv_payload = torch.empty_like(qkv_payload)
                output_qkv_scale = torch.empty_like(qkv_scale)
                payload_work = dist.all_to_all_single(output_qkv_payload, qkv_payload, group=ctx.seq_p_group, async_op=True)
                scale_work = dist.all_to_all_single(output_qkv_scale, qkv_scale, group=ctx.seq_p_group, async_op=True)
                qkv_records.append((local_head, qkv_payload, qkv_scale, output_qkv_payload, output_qkv_scale, qkv_shape, qkv_scale_shape, payload_work, scale_work))
        else:
            if ctx.q_only_img:
                img_qkv_heads = qonly_qkv_pre_heads(q, k, v, ctx.img_start, ctx.img_len, ctx.world_size, ctx.shard_heads)
            else:
                img_qkv_heads = qkv_pre_heads(q, k, v, ctx.img_start, ctx.img_len, ctx.world_size, ctx.shard_heads)
            for local_head in range(ctx.shard_heads):
                img_qkv = img_qkv_heads[local_head]
                output_qkv = torch.empty_like(img_qkv)
                work = dist.all_to_all_single(output_qkv, img_qkv, group=ctx.seq_p_group, async_op=True)
                qkv_records.append((local_head, img_qkv, output_qkv, None, None, work))

        attn_records = []
        for record in qkv_records:
            if ctx.use_fp8_comm:
                local_head, _payload, _scale, output_payload, output_scale, qkv_shape, qkv_scale_shape, payload_work, scale_work = record
                payload_work.wait()
                scale_work.wait()
                if ctx.q_only_img:
                    q_h, k_h, v_h = qonly_qkv_post_head_fp8(
                        output_payload, output_scale, qkv_shape, qkv_scale_shape, k, v, ctx.cur_rank, ctx.txt_start, ctx.txt_storage_len, ctx.img_first, local_head, ctx.shard_heads
                    )
                else:
                    q_h, k_h, v_h = qkv_post_head_fp8(
                        output_payload, output_scale, qkv_shape, qkv_scale_shape, q, k, v, ctx.cur_rank, ctx.txt_start, ctx.txt_storage_len, ctx.img_first, local_head, ctx.shard_heads
                    )
            else:
                local_head, _input_buf, output_buf, qkv_shape, qkv_scale_shape, work = record
                work.wait()
                if ctx.q_only_img:
                    q_h, k_h, v_h = qonly_qkv_post_head(output_buf, k, v, ctx.cur_rank, ctx.txt_start, ctx.txt_storage_len, ctx.img_first, local_head, ctx.shard_heads)
                else:
                    q_h, k_h, v_h = qkv_post_head(output_buf, q, k, v, ctx.cur_rank, ctx.txt_start, ctx.txt_storage_len, ctx.img_first, local_head, ctx.shard_heads)

            head_attn = attention_module.apply(
                q=q_h,
                k=k_h,
                v=v_h,
                cu_seqlens_q=ctx.cu_seqlens_q,
                cu_seqlens_kv=ctx.cu_seqlens_kv,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_kv=ctx.max_seqlen_kv,
                **attention_kwargs,
            )
            head_attn = head_attn.reshape(head_attn.shape[0], -1)
            if head_attn.shape[1] != ctx.hidden_dims:
                raise ValueError(f"head_parallel attention output for local_head={local_head} must flatten to hidden_dims={ctx.hidden_dims}, got {tuple(head_attn.shape)}.")
            if ctx.q_only_img:
                attn_records.append(OptUlyssesAttnWeight._launch_head_parallel_img_attn_a2a(head_attn, ctx))
            else:
                attn_records.append(OptUlyssesAttnWeight._launch_head_parallel_attn_a2a(head_attn, ctx))

        if ctx.q_only_img:
            return OptUlyssesAttnWeight._finish_head_parallel_img_attn_a2a(attn_records, ctx)
        return OptUlyssesAttnWeight._finish_head_parallel_attn_a2a(attn_records, ctx)

    @staticmethod
    def _qkv_a2a(q, k, v, ctx):
        """Run the first Ulysses all-to-all for the default fused q/k/v layout."""
        if ctx.use_fp8_comm:
            qkv_payload, qkv_scale, qkv_shape, qkv_scale_shape = qkv_pre_fp8(q, k, v, ctx.img_start, ctx.img_len, ctx.world_size)
            output_qkv_payload = torch.empty_like(qkv_payload)
            output_qkv_scale = torch.empty_like(qkv_scale)
            dist.all_to_all_single(output_qkv_payload, qkv_payload, group=ctx.seq_p_group)
            dist.all_to_all_single(output_qkv_scale, qkv_scale, group=ctx.seq_p_group)
            return qkv_post_fp8(output_qkv_payload, output_qkv_scale, qkv_shape, qkv_scale_shape, q, k, v, ctx.cur_rank, ctx.txt_start, ctx.txt_storage_len, ctx.img_first)

        img_qkv = qkv_pre(q, k, v, ctx.img_start, ctx.img_len, ctx.world_size)
        output_qkv = torch.empty_like(img_qkv)
        dist.all_to_all_single(output_qkv, img_qkv, group=ctx.seq_p_group)
        return qkv_post(output_qkv, q, k, v, ctx.cur_rank, ctx.txt_start, ctx.txt_storage_len, ctx.img_first)

    @staticmethod
    def _qonly_qkv_a2a(q, k, v, ctx):
        """Run the first Ulysses all-to-all for q_only_img."""
        if ctx.use_fp8_comm:
            qkv_payload, qkv_scale, qkv_shape, qkv_scale_shape = qonly_qkv_pre_fp8(q, k, v, ctx.img_start, ctx.img_len, ctx.world_size)
            output_qkv_payload = torch.empty_like(qkv_payload)
            output_qkv_scale = torch.empty_like(qkv_scale)
            dist.all_to_all_single(output_qkv_payload, qkv_payload, group=ctx.seq_p_group)
            dist.all_to_all_single(output_qkv_scale, qkv_scale, group=ctx.seq_p_group)
            return qonly_qkv_post_fp8(output_qkv_payload, output_qkv_scale, qkv_shape, qkv_scale_shape, k, v, ctx.cur_rank, ctx.txt_start, ctx.txt_storage_len, ctx.img_first)

        img_qkv = qonly_qkv_pre(q, k, v, ctx.img_start, ctx.img_len, ctx.world_size)
        output_qkv = torch.empty_like(img_qkv)
        dist.all_to_all_single(output_qkv, img_qkv, group=ctx.seq_p_group)
        return qonly_qkv_post(output_qkv, k, v, ctx.cur_rank, ctx.txt_start, ctx.txt_storage_len, ctx.img_first)

    @staticmethod
    def _attn_a2a(attn, ctx):
        """Run the second Ulysses all-to-all and merge gathered text output."""
        if ctx.img_first:
            txt_attn = attn[ctx.global_img_len :]
            attn_img_start = 0
        else:
            txt_attn = attn[: ctx.txt_storage_len]
            attn_img_start = ctx.txt_storage_len

        if ctx.use_fp8_comm:
            attn_payload, attn_scale, attn_shape, attn_scale_shape = attn_pre_fp8(attn, attn_img_start, ctx.img_len, ctx.world_size, ctx.shard_heads, ctx.hidden_dims)
            a2a_payload = torch.empty_like(attn_payload)
            a2a_scale = torch.empty_like(attn_scale)
            dist.all_to_all_single(a2a_payload, attn_payload, group=ctx.seq_p_group)
            dist.all_to_all_single(a2a_scale, attn_scale, group=ctx.seq_p_group)
        else:
            a2a_input = attn_pre(attn, attn_img_start, ctx.img_len, ctx.world_size, ctx.shard_heads, ctx.hidden_dims)
            a2a_output = torch.empty_like(a2a_input)
            dist.all_to_all_single(a2a_output, a2a_input, group=ctx.seq_p_group)

        gathered_txt_attn = [torch.empty_like(txt_attn) for _ in range(ctx.world_size)]
        dist.all_gather(gathered_txt_attn, txt_attn, group=ctx.seq_p_group)
        txt_attn = torch.cat(gathered_txt_attn, dim=1)
        if ctx.use_fp8_comm:
            return attn_post_fp8(a2a_payload, a2a_scale, attn_shape, attn_scale_shape, txt_attn, ctx.img_first)
        return attn_post(a2a_output, txt_attn, ctx.img_first)

    @staticmethod
    def _img_attn_a2a(attn, ctx):
        """Run the image-only second Ulysses stage used by q_only_img."""
        if ctx.use_fp8_comm:
            attn_payload, attn_scale, attn_shape, attn_scale_shape = attn_pre_fp8(attn, 0, ctx.img_len, ctx.world_size, ctx.shard_heads, ctx.hidden_dims)
            a2a_payload = torch.empty_like(attn_payload)
            a2a_scale = torch.empty_like(attn_scale)
            dist.all_to_all_single(a2a_payload, attn_payload, group=ctx.seq_p_group)
            dist.all_to_all_single(a2a_scale, attn_scale, group=ctx.seq_p_group)
            txt_attn = attn.new_empty((0, ctx.world_size * ctx.shard_heads * ctx.hidden_dims))
            return attn_post_fp8(a2a_payload, a2a_scale, attn_shape, attn_scale_shape, txt_attn, True)

        a2a_input = attn_pre(attn, 0, ctx.img_len, ctx.world_size, ctx.shard_heads, ctx.hidden_dims)
        a2a_output = torch.empty_like(a2a_input)
        dist.all_to_all_single(a2a_output, a2a_input, group=ctx.seq_p_group)
        txt_attn = attn.new_empty((0, ctx.world_size * ctx.shard_heads * ctx.hidden_dims))
        return attn_post(a2a_output, txt_attn, True)

    @staticmethod
    def _launch_head_parallel_attn_a2a(head_attn, ctx):
        """Launch one local-head image all-to-all as soon as that head is ready."""
        if ctx.img_first:
            txt_attn = head_attn[ctx.global_img_len :]
            attn_img_start = 0
        else:
            txt_attn = head_attn[: ctx.txt_storage_len]
            attn_img_start = ctx.txt_storage_len

        if ctx.use_fp8_comm:
            attn_payload, attn_scale, attn_shape, attn_scale_shape = attn_pre_fp8(head_attn, attn_img_start, ctx.img_len, ctx.world_size, 1, ctx.hidden_dims)
            a2a_payload = torch.empty_like(attn_payload)
            a2a_scale = torch.empty_like(attn_scale)
            payload_work = dist.all_to_all_single(a2a_payload, attn_payload, group=ctx.seq_p_group, async_op=True)
            scale_work = dist.all_to_all_single(a2a_scale, attn_scale, group=ctx.seq_p_group, async_op=True)
            return (txt_attn, attn_payload, attn_scale, a2a_payload, a2a_scale, attn_shape, attn_scale_shape, payload_work, scale_work)

        a2a_input = attn_pre(head_attn, attn_img_start, ctx.img_len, ctx.world_size, 1, ctx.hidden_dims)
        a2a_output = torch.empty_like(a2a_input)
        work = dist.all_to_all_single(a2a_output, a2a_input, group=ctx.seq_p_group, async_op=True)
        return (txt_attn, a2a_input, a2a_output, None, None, work)

    @staticmethod
    def _finish_head_parallel_attn_a2a(records, ctx):
        """Finish launched per-head image all-to-all, then gather text heads."""
        # Do not interleave text all-gather collectives with pending image
        # all-to-all work on the same process group. This keeps the old safe
        # phase boundary while letting image all-to-all be queued earlier.
        for record in records:
            if ctx.use_fp8_comm:
                record[-2].wait()
                record[-1].wait()
            else:
                record[-1].wait()

        local_txt_heads = torch.stack([record[0] for record in records], dim=0)
        if local_txt_heads.shape[1] == 0:
            gathered_txt_heads = None
        else:
            gathered_txt_heads = [torch.empty_like(local_txt_heads) for _ in range(ctx.world_size)]
            dist.all_gather(gathered_txt_heads, local_txt_heads, group=ctx.seq_p_group)

        head_outputs = []
        for local_head, record in enumerate(records):
            if ctx.use_fp8_comm:
                _txt_attn, _payload, _scale, a2a_payload, a2a_scale, attn_shape, attn_scale_shape, _payload_work, _scale_work = record
            else:
                _txt_attn, _input_buf, a2a_output, _unused0, _unused1, _work = record
            if gathered_txt_heads is None:
                txt_attn = local_txt_heads.new_empty((0, ctx.world_size * ctx.hidden_dims))
            else:
                txt_attn = torch.cat([rank_txt[local_head] for rank_txt in gathered_txt_heads], dim=1)
            if ctx.use_fp8_comm:
                head_out = attn_post_fp8(a2a_payload, a2a_scale, attn_shape, attn_scale_shape, txt_attn, ctx.img_first)
            else:
                head_out = attn_post(a2a_output, txt_attn, ctx.img_first)
            head_outputs.append(head_out.reshape(head_out.shape[0], ctx.world_size, ctx.hidden_dims))

        return torch.stack(head_outputs, dim=2).reshape(head_outputs[0].shape[0], ctx.world_size * len(head_outputs) * ctx.hidden_dims)

    @staticmethod
    def _launch_head_parallel_img_attn_a2a(head_attn, ctx):
        """Launch one image-only local-head all-to-all for q_only_img."""
        if ctx.use_fp8_comm:
            attn_payload, attn_scale, attn_shape, attn_scale_shape = attn_pre_fp8(head_attn, 0, ctx.img_len, ctx.world_size, 1, ctx.hidden_dims)
            a2a_payload = torch.empty_like(attn_payload)
            a2a_scale = torch.empty_like(attn_scale)
            payload_work = dist.all_to_all_single(a2a_payload, attn_payload, group=ctx.seq_p_group, async_op=True)
            scale_work = dist.all_to_all_single(a2a_scale, attn_scale, group=ctx.seq_p_group, async_op=True)
            return (head_attn, attn_payload, attn_scale, a2a_payload, a2a_scale, attn_shape, attn_scale_shape, payload_work, scale_work)

        a2a_input = attn_pre(head_attn, 0, ctx.img_len, ctx.world_size, 1, ctx.hidden_dims)
        a2a_output = torch.empty_like(a2a_input)
        work = dist.all_to_all_single(a2a_output, a2a_input, group=ctx.seq_p_group, async_op=True)
        return (head_attn, a2a_input, a2a_output, None, None, work)

    @staticmethod
    def _finish_head_parallel_img_attn_a2a(records, ctx):
        """Finish image-only per-head all-to-all for q_only_img."""
        for record in records:
            if ctx.use_fp8_comm:
                record[-2].wait()
                record[-1].wait()
            else:
                record[-1].wait()

        head_outputs = []
        for record in records:
            if ctx.use_fp8_comm:
                head_attn, _payload, _scale, a2a_payload, a2a_scale, attn_shape, attn_scale_shape, _payload_work, _scale_work = record
            else:
                head_attn, _input_buf, a2a_output, _unused0, _unused1, _work = record
            txt_attn = head_attn.new_empty((0, ctx.world_size * ctx.hidden_dims))
            if ctx.use_fp8_comm:
                head_out = attn_post_fp8(a2a_payload, a2a_scale, attn_shape, attn_scale_shape, txt_attn, True)
            else:
                head_out = attn_post(a2a_output, txt_attn, True)
            head_outputs.append(head_out.reshape(head_out.shape[0], ctx.world_size, ctx.hidden_dims))

        return torch.stack(head_outputs, dim=2).reshape(head_outputs[0].shape[0], ctx.world_size * len(head_outputs) * ctx.hidden_dims)

    @staticmethod
    def _build_context(
        world_size,
        cur_rank,
        img_len,
        txt_len,
        txt_mask_len,
        q_heads,
        hidden_dims,
        img_first,
        q_only_img,
        use_fp8_comm,
        seq_p_group,
    ):
        global_img_len = img_len * world_size
        txt_storage_len = txt_mask_len if txt_mask_len is not None else txt_len
        cu_seqlens_kv = OptUlyssesAttnWeight._build_attention_cu_seqlens(txt_len, txt_mask_len, global_img_len)
        max_seqlen_kv = global_img_len + txt_storage_len
        if q_only_img:
            cu_seqlens_q = OptUlyssesAttnWeight._build_img_only_cu_seqlens(global_img_len)
            max_seqlen_q = global_img_len
        else:
            cu_seqlens_q = cu_seqlens_kv
            max_seqlen_q = max_seqlen_kv

        return _UlyssesContext(
            world_size=world_size,
            cur_rank=cur_rank,
            img_len=img_len,
            txt_len=txt_len,
            txt_mask_len=txt_mask_len,
            shard_heads=q_heads // world_size,
            hidden_dims=hidden_dims,
            global_img_len=global_img_len,
            img_start=0 if img_first else txt_storage_len,
            txt_start=img_len if img_first else 0,
            img_first=img_first,
            q_only_img=q_only_img,
            use_fp8_comm=use_fp8_comm,
            seq_p_group=seq_p_group,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
        )

    @staticmethod
    def _build_attention_cu_seqlens(txt_len, txt_mask_len, global_img_len):
        cu_seqlens = torch.zeros([2], dtype=torch.int32)
        cu_seqlens[1] = txt_len + global_img_len
        if txt_mask_len is not None:
            cu_seqlens = torch.cat((cu_seqlens, torch.tensor([txt_mask_len + global_img_len], dtype=torch.int32)))
        return cu_seqlens

    @staticmethod
    def _build_img_only_cu_seqlens(global_img_len):
        cu_seqlens = torch.zeros([2], dtype=torch.int32)
        cu_seqlens[1] = global_img_len
        return cu_seqlens

    @staticmethod
    def _as_int(value):
        if isinstance(value, torch.Tensor):
            return int(value.item())
        return int(value)

    @staticmethod
    def _parse_lengths(slice_qkv_len, cu_seqlens_qkv, img_first):
        if img_first:
            img_len = OptUlyssesAttnWeight._as_int(slice_qkv_len)
            txt_len = OptUlyssesAttnWeight._as_int(cu_seqlens_qkv[1]) - img_len
            txt_mask_len = OptUlyssesAttnWeight._as_int(cu_seqlens_qkv[2]) - img_len if len(cu_seqlens_qkv) == 3 else None
        else:
            txt_len = OptUlyssesAttnWeight._as_int(slice_qkv_len)
            img_len = OptUlyssesAttnWeight._as_int(cu_seqlens_qkv[1]) - txt_len
            txt_mask_len = None
        return img_len, txt_len, txt_mask_len
