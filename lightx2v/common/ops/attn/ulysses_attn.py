from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate
from .ulysses_a2a import create_ulysses_a2a_backend
from .ulysses_prepost import create_ulysses_prepost_backend
from .utils.seq_p import flatten_seq_p_tensor, split_main_aux_output, validate_seq_p_inputs


@dataclass
class _HeadGroupExchange:
    """Keep group inputs and packed send buffers alive until exchanges finish."""

    size: int
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    aux_q: torch.Tensor | None
    aux_k: torch.Tensor | None
    aux_v: torch.Tensor | None
    head_index: int | None
    packed: tuple
    exchanged: tuple
    works: tuple


@ATTN_WEIGHT_REGISTER("ulysses")
class UlyssesAttnWeight(AttnWeightTemplate):
    """Composable Ulysses attention with independent layout and A2A backends."""

    def __init__(self, a2a_backend="torch"):
        self.config = {}
        self.default_a2a_backend = a2a_backend

    def apply(
        self,
        q,
        k,
        v,
        aux_q=None,
        aux_k=None,
        aux_v=None,
        attention_module=None,
        seq_p_group=None,
        prepost_backend=None,
        a2a_backend=None,
        quant_scheme=None,
        tensor_fusion=False,
        head_parallel=False,
        aux_first=False,
        attention_kwargs=None,
        head_parallel_group_size=1,
    ):
        """Run Ulysses attention over QKV and optional auxiliary token regions.

        ``q``, ``k``, and ``v`` contain this rank's local sequence partition with
        all heads and are exchanged by Ulysses A2A. Every rank must provide the
        same local sequence length; the caller is responsible for padding and
        partitioning before this hot-path interface. ``aux_*`` contain optional
        tokens available in full on every sequence-parallel rank; they bypass QKV
        A2A and are sliced by head locally. Passing all auxiliary tensors as
        ``None`` selects QKV-only attention; passing only ``aux_q=None`` selects
        the q-only cross-attention form. Set ``aux_first=True`` when auxiliary
        tokens precede the A2A tokens in the attention sequence.
        This interface supports one logical Q sequence and one logical KV
        sequence only. Packed-varlen batches are not supported.
        ``tensor_fusion`` packs Q/K/V into one Ulysses communication payload.
        ``head_parallel_group_size`` controls how many local attention heads are
        processed in each pipelined stage when ``head_parallel`` is enabled.
        The return value is ``(output, aux_output)``.
        """
        q = flatten_seq_p_tensor(q, "q")
        k = flatten_seq_p_tensor(k, "k")
        v = flatten_seq_p_tensor(v, "v")
        aux_q = flatten_seq_p_tensor(aux_q, "aux_q")
        aux_k = flatten_seq_p_tensor(aux_k, "aux_k")
        aux_v = flatten_seq_p_tensor(aux_v, "aux_v")
        if attention_kwargs is None:
            attention_kwargs = {}

        world_size = dist.get_world_size(seq_p_group)
        rank = dist.get_rank(seq_p_group)
        self._validate_inputs(q, k, v, aux_q, aux_k, aux_v, world_size)

        qkv_fusion = tensor_fusion
        prepost_name, a2a_name, qkv_fusion = self._resolve_options(
            prepost_backend,
            a2a_backend,
            q,
            k,
            quant_scheme,
            qkv_fusion,
            head_parallel,
        )
        prepost = create_ulysses_prepost_backend(prepost_name)
        a2a = create_ulysses_a2a_backend(a2a_name)

        common_args = dict(
            prepost=prepost,
            a2a=a2a,
            q=q,
            k=k,
            v=v,
            aux_q=aux_q,
            aux_k=aux_k,
            aux_v=aux_v,
            attention_module=attention_module,
            seq_p_group=seq_p_group,
            quant_scheme=quant_scheme,
            qkv_fusion=qkv_fusion,
            aux_first=aux_first,
            world_size=world_size,
            rank=rank,
            attention_kwargs=attention_kwargs,
        )
        if head_parallel:
            return self._apply_head_pipeline(head_group_size=head_parallel_group_size, **common_args)
        return self._apply_bulk(**common_args)

    @staticmethod
    def _apply_bulk(
        prepost,
        a2a,
        q,
        k,
        v,
        aux_q,
        aux_k,
        aux_v,
        attention_module,
        seq_p_group,
        quant_scheme,
        qkv_fusion,
        aux_first,
        world_size,
        rank,
        attention_kwargs,
    ):
        local_len, q_heads, hidden_dims = q.shape
        shard_heads = q_heads // world_size
        global_len = local_len * world_size
        aux_len = 0 if aux_q is None else aux_q.shape[0]

        packed_qkv = prepost.pack_qkv(q, k, v, world_size, quant_scheme, qkv_fusion)
        exchanged_qkv = UlyssesAttnWeight._exchange_packed(packed_qkv, a2a, seq_p_group)
        attn_q, attn_k, attn_v = prepost.unpack_qkv(
            exchanged_qkv,
            q,
            k,
            v,
            aux_q,
            aux_k,
            aux_v,
            rank,
            world_size,
            aux_first,
        )
        dense_attention_kwargs = UlyssesAttnWeight._dense_attention_kwargs(attention_kwargs, attn_q, attn_k)

        attn = attention_module.apply(
            q=attn_q,
            k=attn_k,
            v=attn_v,
            **dense_attention_kwargs,
        ).reshape(attn_q.shape[0], -1)
        output, local_aux_attn = split_main_aux_output(attn, global_len, aux_len, aux_first)

        packed_attn = prepost.pack_attn(output, local_len, world_size, shard_heads, hidden_dims, quant_scheme)
        exchanged_attn = UlyssesAttnWeight._exchange_packed(packed_attn, a2a, seq_p_group)
        output = prepost.unpack_attn(exchanged_attn, attn.dtype, hidden_dims)

        aux_output = UlyssesAttnWeight._gather_aux(local_aux_attn, world_size, seq_p_group)
        return output, aux_output

    @staticmethod
    def _apply_head_pipeline(
        prepost,
        a2a,
        q,
        k,
        v,
        aux_q,
        aux_k,
        aux_v,
        attention_module,
        seq_p_group,
        quant_scheme,
        qkv_fusion,
        aux_first,
        world_size,
        rank,
        attention_kwargs,
        head_group_size,
    ):
        local_len, q_heads, hidden_dims = q.shape
        shard_heads = q_heads // world_size
        global_len = local_len * world_size
        aux_len = 0 if aux_q is None else aux_q.shape[0]
        if not isinstance(head_group_size, int) or isinstance(head_group_size, bool):
            raise TypeError(f"head_parallel_group_size must be an integer, got {type(head_group_size).__name__}.")
        if not 1 <= head_group_size <= shard_heads:
            raise ValueError(f"head_parallel_group_size must be in [1, {shard_heads}], got {head_group_size}.")

        def select_group(tensor, begin, end):
            if tensor is None:
                return None
            heads_by_rank = tensor.reshape(tensor.shape[0], world_size, shard_heads, hidden_dims)
            group = heads_by_rank[:, :, begin:end]
            return group.reshape(tensor.shape[0], world_size * (end - begin), hidden_dims).contiguous()

        qkv_records = []
        for begin in range(0, shard_heads, head_group_size):
            end = min(begin + head_group_size, shard_heads)
            group_size = end - begin
            if group_size == 1:
                # Reuse the original tensors for singleton groups. Both pre/post
                # backends can select one interleaved head without retaining a
                # full-size contiguous Q/K/V copy for every pipeline stage.
                group_q, group_k, group_v = q, k, v
                group_aux_q, group_aux_k, group_aux_v = aux_q, aux_k, aux_v
                head_index = begin
            else:
                group_q = select_group(q, begin, end)
                group_k = select_group(k, begin, end)
                group_v = select_group(v, begin, end)
                group_aux_q = select_group(aux_q, begin, end)
                group_aux_k = select_group(aux_k, begin, end)
                group_aux_v = select_group(aux_v, begin, end)
                head_index = None
            packed = prepost.pack_qkv(
                group_q,
                group_k,
                group_v,
                world_size,
                quant_scheme,
                qkv_fusion,
                head_index=head_index,
            )
            exchanged, works = UlyssesAttnWeight._exchange_packed_async(packed, a2a, seq_p_group)
            qkv_records.append(
                _HeadGroupExchange(
                    size=group_size,
                    q=group_q,
                    k=group_k,
                    v=group_v,
                    aux_q=group_aux_q,
                    aux_k=group_aux_k,
                    aux_v=group_aux_v,
                    head_index=head_index,
                    packed=packed,
                    exchanged=exchanged,
                    works=works,
                )
            )

        attn_records = []
        local_aux_heads = []
        for record in qkv_records:
            group_size = record.size
            UlyssesAttnWeight._wait_works(record.works)
            attn_q, attn_k, attn_v = prepost.unpack_qkv(
                record.exchanged,
                record.q,
                record.k,
                record.v,
                record.aux_q,
                record.aux_k,
                record.aux_v,
                rank,
                world_size,
                aux_first,
                head_index=record.head_index,
            )
            dense_attention_kwargs = UlyssesAttnWeight._dense_attention_kwargs(attention_kwargs, attn_q, attn_k)
            group_attn = attention_module.apply(
                q=attn_q,
                k=attn_k,
                v=attn_v,
                **dense_attention_kwargs,
            ).reshape(attn_q.shape[0], -1)
            expected_width = group_size * hidden_dims
            if group_attn.shape[1] != expected_width:
                raise ValueError(f"head_parallel attention output must flatten to {expected_width}, got shape={tuple(group_attn.shape)}.")

            output, local_aux_attn = split_main_aux_output(group_attn, global_len, aux_len, aux_first)
            if local_aux_attn is not None:
                local_aux_heads.append(local_aux_attn.reshape(local_aux_attn.shape[0], group_size, hidden_dims))

            packed_attn = prepost.pack_attn(output, local_len, world_size, group_size, hidden_dims, quant_scheme)
            exchanged_attn, attn_works = UlyssesAttnWeight._exchange_packed_async(packed_attn, a2a, seq_p_group)
            attn_records.append((group_size, packed_attn, exchanged_attn, attn_works, group_attn.dtype))

        group_outputs = []
        for group_size, _, exchanged_attn, attn_works, output_dtype in attn_records:
            UlyssesAttnWeight._wait_works(attn_works)
            group_output = prepost.unpack_attn(exchanged_attn, output_dtype, hidden_dims)
            group_outputs.append(group_output.reshape(group_output.shape[0], world_size, group_size, hidden_dims))
        output = torch.cat(group_outputs, dim=2).reshape(group_outputs[0].shape[0], -1)

        if local_aux_heads:
            local_aux = torch.cat(local_aux_heads, dim=1)
            aux_output = UlyssesAttnWeight._gather_aux(local_aux, world_size, seq_p_group).reshape(local_aux.shape[0], -1)
        else:
            aux_output = None
        return output, aux_output

    @staticmethod
    def _exchange_packed(packed, a2a, group):
        exchanged = []
        for payload, scale in packed:
            output_payload, _ = a2a.exchange(payload, group=group, async_op=False)
            output_scale = None
            if scale is not None:
                output_scale, _ = a2a.exchange(scale, group=group, async_op=False)
            exchanged.append((output_payload, output_scale))
        return tuple(exchanged)

    @staticmethod
    def _exchange_packed_async(packed, a2a, group):
        exchanged = []
        works = []
        for payload, scale in packed:
            output_payload, payload_work = a2a.exchange(payload, group=group, async_op=True)
            output_scale = None
            scale_work = None
            if scale is not None:
                output_scale, scale_work = a2a.exchange(scale, group=group, async_op=True)
            exchanged.append((output_payload, output_scale))
            works.extend((payload_work, scale_work))
        return tuple(exchanged), tuple(work for work in works if work is not None)

    @staticmethod
    def _wait_works(works):
        for work in works:
            work.wait()

    @staticmethod
    def _dense_attention_kwargs(attention_kwargs, q, k):
        dense_kwargs = dict(attention_kwargs)
        dense_kwargs.setdefault("cu_seqlens_q", None)
        dense_kwargs.setdefault("cu_seqlens_kv", None)
        dense_kwargs.setdefault("max_seqlen_q", q.shape[0])
        dense_kwargs.setdefault("max_seqlen_kv", k.shape[0])
        return dense_kwargs

    @staticmethod
    def _gather_aux(local_aux, world_size, group):
        if local_aux is None:
            return None
        gathered = [torch.empty_like(local_aux) for _ in range(world_size)]
        dist.all_gather(gathered, local_aux, group=group)
        return torch.cat(gathered, dim=1)

    def _resolve_options(self, prepost_backend, a2a_backend, q, k, quant_scheme, qkv_fusion, head_parallel):
        prepost_name = "torch" if prepost_backend is None else prepost_backend
        a2a_name = self.default_a2a_backend if a2a_backend is None else a2a_backend
        is_gqa = q.shape[1] != k.shape[1]
        if prepost_name == "triton":
            if q.device.type != "cuda":
                raise ValueError("prepost_backend='triton' requires CUDA tensors.")
            if is_gqa:
                raise ValueError("prepost_backend='triton' does not support GQA; select prepost_backend='torch'.")
            if quant_scheme == "fp4":
                raise ValueError("prepost_backend='triton' does not support FP4 communication; select prepost_backend='torch'.")
            if not qkv_fusion:
                raise ValueError("prepost_backend='triton' requires tensor_fusion=True.")

        if head_parallel and is_gqa:
            raise ValueError("head_parallel does not support GQA.")
        if qkv_fusion and is_gqa:
            raise ValueError("tensor_fusion does not support GQA.")
        if head_parallel and a2a_name == "round_robin":
            raise ValueError("a2a_backend='round_robin' is incompatible with head_parallel=True.")

        return prepost_name, a2a_name, qkv_fusion

    @staticmethod
    def _validate_inputs(q, k, v, aux_q, aux_k, aux_v, world_size):
        validate_seq_p_inputs(q, k, v, aux_q, aux_k, aux_v)
        if q.shape[1] % world_size or k.shape[1] % world_size:
            raise ValueError(f"q_heads and kv_heads must be divisible by world_size={world_size}.")


ATTN_WEIGHT_REGISTER.register(
    partial(UlyssesAttnWeight, a2a_backend="round_robin"),
    key="ulysses-4090",
)
