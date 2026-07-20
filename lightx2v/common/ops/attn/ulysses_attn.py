from functools import partial

import torch
import torch.distributed as dist

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate
from .ulysses_a2a import create_ulysses_a2a_backend
from .ulysses_prepost import create_ulysses_prepost_backend


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
        """Deprecated compatibility adapter for the legacy joined-token interface.

        New model integrations should call :meth:`apply_new`
        with QKV and optional auxiliary token regions passed separately.
        ``cu_seqlens_qkv`` is retained only for call-signature compatibility;
        its value is ignored.
        """
        q = self._flatten_tensor(q)
        k = self._flatten_tensor(k)
        v = self._flatten_tensor(v)

        split_len = int(slice_qkv_len.item()) if isinstance(slice_qkv_len, torch.Tensor) else int(slice_qkv_len)
        main_first = img_first
        main_q_only = q_only_img
        if main_first:
            main_len, aux_len = split_len, k.shape[0] - split_len
        else:
            main_len, aux_len = k.shape[0] - split_len, split_len
        if main_q_only:
            q = q.contiguous()
            aux_q = None
            k, aux_k = self._legacy_split_tensor(k, main_len, aux_len, main_first)
            v, aux_v = self._legacy_split_tensor(v, main_len, aux_len, main_first)
        else:
            q, aux_q = self._legacy_split_tensor(q, main_len, aux_len, main_first)
            k, aux_k = self._legacy_split_tensor(k, main_len, aux_len, main_first)
            v, aux_v = self._legacy_split_tensor(v, main_len, aux_len, main_first)

        if use_fp8_comm and use_fp4_comm:
            raise ValueError("use_fp8_comm and use_fp4_comm cannot both be enabled.")
        quant_scheme = "fp8" if use_fp8_comm else "fp4" if use_fp4_comm else None
        output, aux_attn = self.apply_new(
            q=q,
            k=k,
            v=v,
            aux_q=aux_q,
            aux_k=aux_k,
            aux_v=aux_v,
            attention_module=attention_module,
            seq_p_group=seq_p_group,
            quant_scheme=quant_scheme,
            qkv_fusion=use_tensor_fusion,
            head_parallel=enable_head_parallel,
            aux_first=not main_first,
            attention_kwargs=kwargs,
        )
        return self._legacy_join_output(output, aux_attn, main_first)

    def apply_new(
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
        qkv_fusion=False,
        head_parallel=False,
        aux_first=False,
        attention_kwargs=None,
    ):
        """Run Ulysses attention over QKV and optional auxiliary token regions.

        ``q``, ``k``, and ``v`` contain this rank's local sequence partition with
        all heads and are exchanged by Ulysses A2A. ``aux_*`` contain optional
        tokens available in full on every sequence-parallel rank; they bypass QKV
        A2A and are sliced by head locally. Passing all auxiliary tensors as
        ``None`` selects QKV-only attention; passing only ``aux_q=None`` selects
        the q-only cross-attention form. Set ``aux_first=True`` when auxiliary
        tokens precede the A2A tokens in the attention sequence.
        This interface supports one logical Q sequence and one logical KV
        sequence only. Packed-varlen batches are not supported.
        The return value is ``(output, aux_output)``.
        """
        q = self._flatten_tensor(q)
        k = self._flatten_tensor(k)
        v = self._flatten_tensor(v)
        aux_q = self._flatten_tensor(aux_q)
        aux_k = self._flatten_tensor(aux_k)
        aux_v = self._flatten_tensor(aux_v)
        if attention_kwargs is None:
            attention_kwargs = {}

        world_size = dist.get_world_size(seq_p_group)
        rank = dist.get_rank(seq_p_group)
        self._validate_inputs(q, k, v, aux_q, aux_k, aux_v, world_size)
        self._validate_sequence_partition(q.shape[0], world_size, seq_p_group, q.device)

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
            return self._apply_head_pipeline(**common_args)
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

        attn = attention_module.apply(
            q=attn_q,
            k=attn_k,
            v=attn_v,
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            max_seqlen_q=None,
            max_seqlen_kv=None,
            **attention_kwargs,
        ).reshape(attn_q.shape[0], -1)
        output, local_aux_attn = UlyssesAttnWeight._split_attention_output(attn, global_len, aux_q, aux_first)

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
    ):
        local_len, q_heads, hidden_dims = q.shape
        shard_heads = q_heads // world_size
        global_len = local_len * world_size

        qkv_records = []
        for head_index in range(shard_heads):
            packed = prepost.pack_qkv(
                q,
                k,
                v,
                world_size,
                quant_scheme,
                qkv_fusion,
                head_index=head_index,
            )
            exchanged, works = UlyssesAttnWeight._exchange_packed_async(packed, a2a, seq_p_group)
            qkv_records.append((packed, exchanged, works))

        attn_records = []
        local_aux_heads = []
        for head_index, (_, exchanged_qkv, qkv_works) in enumerate(qkv_records):
            UlyssesAttnWeight._wait_works(qkv_works)
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
                head_index=head_index,
            )
            head_attn = attention_module.apply(
                q=attn_q,
                k=attn_k,
                v=attn_v,
                cu_seqlens_q=None,
                cu_seqlens_kv=None,
                max_seqlen_q=None,
                max_seqlen_kv=None,
                **attention_kwargs,
            ).reshape(attn_q.shape[0], -1)
            if head_attn.shape[1] != hidden_dims:
                raise ValueError(f"head_parallel attention output must flatten to hidden_dims={hidden_dims}, got shape={tuple(head_attn.shape)}.")

            output, local_aux_attn = UlyssesAttnWeight._split_attention_output(head_attn, global_len, aux_q, aux_first)
            if local_aux_attn is not None:
                local_aux_heads.append(local_aux_attn.reshape(local_aux_attn.shape[0], 1, hidden_dims))

            packed_attn = prepost.pack_attn(output, local_len, world_size, 1, hidden_dims, quant_scheme)
            exchanged_attn, attn_works = UlyssesAttnWeight._exchange_packed_async(packed_attn, a2a, seq_p_group)
            attn_records.append((packed_attn, exchanged_attn, attn_works, head_attn.dtype))

        head_outputs = []
        for _, exchanged_attn, attn_works, output_dtype in attn_records:
            UlyssesAttnWeight._wait_works(attn_works)
            head_output = prepost.unpack_attn(exchanged_attn, output_dtype, hidden_dims)
            head_outputs.append(head_output.reshape(head_output.shape[0], world_size, hidden_dims))
        output = torch.stack(head_outputs, dim=2).reshape(head_outputs[0].shape[0], -1)

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
    def _split_attention_output(attn, global_len, aux_q, aux_first):
        if aux_q is None:
            return attn, None
        if aux_first:
            return attn[-global_len:], attn[:-global_len]
        return attn[:global_len], attn[global_len:]

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
                raise ValueError("prepost_backend='triton' requires qkv_fusion=True.")

        if head_parallel and is_gqa:
            raise ValueError("head_parallel does not support GQA.")
        if qkv_fusion and is_gqa:
            raise ValueError("qkv_fusion does not support GQA.")
        if head_parallel and a2a_name == "round_robin":
            raise ValueError("a2a_backend='round_robin' is incompatible with head_parallel=True.")

        return prepost_name, a2a_name, qkv_fusion

    @staticmethod
    def _validate_inputs(q, k, v, aux_q, aux_k, aux_v, world_size):
        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            raise ValueError("q/k/v must be 3D [sequence, heads, head_dim] tensors.")
        if q.shape[0] <= 0:
            raise ValueError("q/k/v must contain at least one local sequence token.")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError("q/k/v must have the same sequence length.")
        if k.shape != v.shape:
            raise ValueError("k/v must have the same shape.")
        if q.shape[-1] != k.shape[-1]:
            raise ValueError("q/k/v must have the same head_dim.")
        if q.shape[1] % world_size or k.shape[1] % world_size:
            raise ValueError(f"q_heads and kv_heads must be divisible by world_size={world_size}.")

        aux = (aux_q, aux_k, aux_v)
        if (aux_k is None) != (aux_v is None):
            raise ValueError("aux k/v must either both be tensors or both be None.")
        if aux_q is not None and aux_k is None:
            raise ValueError("aux q requires aux k/v.")
        for name, tensor, source in zip(("q", "k", "v"), aux, (q, k, v)):
            if tensor is None:
                continue
            if tensor.dim() != 3 or tensor.shape[1:] != source.shape[1:]:
                raise ValueError(f"aux_{name} must match {name} head dimensions, got {tuple(tensor.shape)} vs {tuple(source.shape)}.")
        if aux_k is not None and aux_k.shape[0] != aux_v.shape[0]:
            raise ValueError("aux k/v must have the same sequence length.")
        if aux_q is not None and aux_q.shape[0] != aux_k.shape[0]:
            raise ValueError("aux q/k/v must have the same sequence length.")

    @staticmethod
    def _flatten_tensor(tensor):
        if tensor is None:
            return None
        if tensor.dim() == 4:
            if tensor.shape[0] != 1:
                raise ValueError(
                    "Ulysses supports one logical sequence only; 4D inputs must have batch size 1, "
                    f"got shape={tuple(tensor.shape)}."
                )
            return tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])
        return tensor

    @staticmethod
    def _validate_sequence_partition(local_len, world_size, group, device):
        """Ensure every sequence-parallel rank owns the same local token count."""
        if world_size == 1:
            return

        local_len_tensor = torch.tensor([local_len], dtype=torch.int64, device=device)
        gathered_lengths = [torch.empty_like(local_len_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_lengths, local_len_tensor, group=group)
        lengths = tuple(int(length.item()) for length in gathered_lengths)
        if len(set(lengths)) != 1:
            raise ValueError(
                "Ulysses requires the same local sequence length on every sequence-parallel rank, "
                f"got lengths={lengths}."
            )

    @staticmethod
    def _legacy_split_tensor(tensor, main_len, aux_len, main_first):
        if main_first:
            return tensor[:main_len].contiguous(), tensor[main_len : main_len + aux_len].contiguous()
        return tensor[aux_len : aux_len + main_len].contiguous(), tensor[:aux_len].contiguous()

    @staticmethod
    def _legacy_join_output(output, aux_output, main_first):
        if aux_output is None or aux_output.shape[0] == 0:
            return output
        if main_first:
            return torch.cat((output, aux_output), dim=0)
        return torch.cat((aux_output, output), dim=0)


ATTN_WEIGHT_REGISTER.register(
    partial(UlyssesAttnWeight, a2a_backend="round_robin"),
    key="ulysses-4090",
)
