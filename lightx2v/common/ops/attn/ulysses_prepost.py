import torch

from lightx2v.utils.quant_utils import dequant_fp8_vllm, quant_fp8_vllm

from .kernels.ulysses_layout import (
    attn_post,
    attn_post_fp8,
    attn_pre,
    attn_pre_fp8,
    qkv_post,
    qkv_post_fp8,
    qkv_pre,
    qkv_pre_fp8,
)

try:
    from sageattn3_sparse import dequant_fp4 as dequant_fp4_sage3
    from sageattn3_sparse import quant_fp4 as quant_fp4_sage3
except ImportError:
    quant_fp4_sage3 = None
    dequant_fp4_sage3 = None


def _validate_quant_scheme(quant_scheme):
    if quant_scheme not in (None, "fp8", "fp4"):
        raise ValueError(f"Unknown quant_scheme={quant_scheme!r}; expected None, 'fp8', or 'fp4'.")
    if quant_scheme != "fp4":
        return
    if quant_fp4_sage3 is None or dequant_fp4_sage3 is None:
        raise ImportError("sageattn3_sparse quant_fp4/dequant_fp4 is required for Ulysses FP4 communication.")


def _pack_tensor(tensor, quant_scheme):
    tensor = tensor.contiguous()
    _validate_quant_scheme(quant_scheme)
    if quant_scheme is None:
        return tensor, None

    shape = tensor.shape
    hidden_dims = shape[-1]
    if quant_scheme == "fp8":
        payload, scale = quant_fp8_vllm(tensor.reshape(-1, hidden_dims))
        return payload.reshape(shape).contiguous(), scale.reshape(*shape[:-1], 1).contiguous()

    if hidden_dims % 16 != 0:
        raise ValueError(f"Ulysses FP4 communication requires hidden_dims divisible by 16, got {hidden_dims}.")
    payload, scale = quant_fp4_sage3(tensor.reshape(1, 1, -1, hidden_dims))
    return payload.reshape(*shape[:-1], hidden_dims // 2).contiguous(), scale.reshape(*shape[:-1], hidden_dims // 16).contiguous()


def _unpack_tensor(packed, output_dtype, hidden_dims):
    payload, scale = packed
    if scale is None:
        return payload
    if payload.dtype == torch.float8_e4m3fn:
        return dequant_fp8_vllm(payload, scale, output_dtype)
    output_shape = (*payload.shape[:-1], hidden_dims)
    return dequant_fp4_sage3(
        payload.reshape(1, 1, -1, hidden_dims // 2),
        scale.reshape(1, 1, -1, hidden_dims // 16),
    ).reshape(output_shape)


def _join_tokens(tensor, aux, aux_first):
    if aux is None or aux.shape[0] == 0:
        return tensor.contiguous()
    if aux_first:
        return torch.cat((aux, tensor), dim=0)
    return torch.cat((tensor, aux), dim=0)


def _select_aux_heads(tensor, rank, head_count, head_index=None):
    if tensor is None:
        return None
    start = rank * head_count
    if head_index is not None:
        start += head_index
        head_count = 1
    return tensor[:, start : start + head_count, :]


class TorchUlyssesPrePost:
    """PyTorch layout and communication-quantization backend."""

    @staticmethod
    def pack_qkv(q, k, v, world_size, quant_scheme=None, qkv_fusion=False, head_index=None):
        local_len, q_heads, hidden_dims = q.shape
        kv_heads = k.shape[1]
        q_shard_heads = q_heads // world_size
        kv_shard_heads = kv_heads // world_size

        if head_index is not None:
            if q.shape != k.shape or q.shape != v.shape:
                raise ValueError("single-head pack requires equal q/k/v head counts.")
            if not 0 <= head_index < q_shard_heads:
                raise ValueError(f"head_index must be in [0, {q_shard_heads}), got {head_index}.")
            q = q.reshape(local_len, world_size, q_shard_heads, hidden_dims)
            k = k.reshape(local_len, world_size, q_shard_heads, hidden_dims)
            v = v.reshape(local_len, world_size, q_shard_heads, hidden_dims)
            head_q = q[:, :, head_index].transpose(0, 1)
            head_k = k[:, :, head_index].transpose(0, 1)
            head_v = v[:, :, head_index].transpose(0, 1)
            if qkv_fusion:
                qkv = torch.stack((head_q, head_k, head_v), dim=2).unsqueeze(3)
                return (_pack_tensor(qkv, quant_scheme),)
            return (
                _pack_tensor(head_q.unsqueeze(2), quant_scheme),
                _pack_tensor(head_k.unsqueeze(2), quant_scheme),
                _pack_tensor(head_v.unsqueeze(2), quant_scheme),
            )

        if qkv_fusion and q.shape == k.shape == v.shape:
            q = q.reshape(local_len, world_size, q_shard_heads, hidden_dims)
            k = k.reshape(local_len, world_size, q_shard_heads, hidden_dims)
            v = v.reshape(local_len, world_size, q_shard_heads, hidden_dims)
            qkv = torch.stack((q, k, v), dim=2).permute(1, 0, 2, 3, 4).contiguous()
            return (_pack_tensor(qkv, quant_scheme),)

        q = q.reshape(local_len, world_size, q_shard_heads, hidden_dims).permute(1, 0, 2, 3).contiguous()
        k = k.reshape(local_len, world_size, kv_shard_heads, hidden_dims).permute(1, 0, 2, 3).contiguous()
        v = v.reshape(local_len, world_size, kv_shard_heads, hidden_dims).permute(1, 0, 2, 3).contiguous()
        return _pack_tensor(q, quant_scheme), _pack_tensor(k, quant_scheme), _pack_tensor(v, quant_scheme)

    @staticmethod
    def unpack_qkv(
        packed,
        q,
        k,
        v,
        aux_q,
        aux_k,
        aux_v,
        rank,
        world_size,
        aux_first,
        head_index=None,
    ):
        hidden_dims = q.shape[-1]
        q_shard_heads = q.shape[1] // world_size
        kv_shard_heads = k.shape[1] // world_size
        if head_index is not None:
            q_shard_heads = kv_shard_heads = 1

        if len(packed) == 1:
            qkv = _unpack_tensor(packed[0], q.dtype, hidden_dims)
            qkv = qkv.reshape(-1, 3, q_shard_heads, hidden_dims)
            global_q, global_k, global_v = qkv.unbind(dim=1)
        else:
            global_q = _unpack_tensor(packed[0], q.dtype, hidden_dims).reshape(-1, q_shard_heads, hidden_dims)
            global_k = _unpack_tensor(packed[1], k.dtype, hidden_dims).reshape(-1, kv_shard_heads, hidden_dims)
            global_v = _unpack_tensor(packed[2], v.dtype, hidden_dims).reshape(-1, kv_shard_heads, hidden_dims)

        local_q_heads = q.shape[1] // world_size
        local_kv_heads = k.shape[1] // world_size
        local_aux_q = _select_aux_heads(aux_q, rank, local_q_heads, head_index)
        local_aux_k = _select_aux_heads(aux_k, rank, local_kv_heads, head_index)
        local_aux_v = _select_aux_heads(aux_v, rank, local_kv_heads, head_index)

        q = _join_tokens(global_q, local_aux_q, aux_first)
        k = _join_tokens(global_k, local_aux_k, aux_first)
        v = _join_tokens(global_v, local_aux_v, aux_first)
        return q, k, v

    @staticmethod
    def pack_attn(output, local_len, world_size, shard_heads, hidden_dims, quant_scheme=None):
        attn = output.reshape(world_size, local_len, shard_heads, hidden_dims).transpose(1, 2).contiguous()
        return (_pack_tensor(attn, quant_scheme),)

    @staticmethod
    def unpack_attn(packed, output_dtype, hidden_dims):
        attn = _unpack_tensor(packed[0], output_dtype, hidden_dims)
        world_size, shard_heads, local_len, _ = attn.shape
        return attn.reshape(world_size * shard_heads, local_len, hidden_dims).transpose(0, 1).contiguous().reshape(local_len, -1)


class TritonUlyssesPrePost:
    """Fused Triton layout and FP8 communication-quantization backend."""

    @staticmethod
    def _validate(q, k, v, quant_scheme):
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("prepost_backend='triton' requires equal q/k/v head counts.")
        if quant_scheme == "fp4":
            raise ValueError("prepost_backend='triton' does not support FP4 communication.")
        _validate_quant_scheme(quant_scheme)

    @classmethod
    def pack_qkv(cls, q, k, v, world_size, quant_scheme=None, qkv_fusion=True, head_index=None):
        cls._validate(q, k, v, quant_scheme)
        if quant_scheme == "fp8":
            payload, scale, _, _ = qkv_pre_fp8(q, k, v, world_size, head_index=head_index)
            return ((payload, scale),)
        return ((qkv_pre(q, k, v, world_size, head_index=head_index), None),)

    @staticmethod
    def unpack_qkv(
        packed,
        q,
        k,
        v,
        aux_q,
        aux_k,
        aux_v,
        rank,
        world_size,
        aux_first,
        head_index=None,
    ):
        payload, scale = packed[0]
        aux_len = 0 if aux_k is None else aux_k.shape[0]
        q_source = q if aux_q is None else aux_q
        k_source = k if aux_k is None else aux_k
        v_source = v if aux_v is None else aux_v
        qkv_first = not aux_first
        q_only = aux_q is None and aux_k is not None

        if scale is None:
            return qkv_post(payload, q_source, k_source, v_source, rank, aux_len, qkv_first, q_only=q_only, head_index=head_index)
        return qkv_post_fp8(
            payload,
            scale,
            payload.shape,
            scale.shape,
            q_source,
            k_source,
            v_source,
            rank,
            aux_len,
            qkv_first,
            q_only=q_only,
            head_index=head_index,
        )

    @staticmethod
    def pack_attn(output, local_len, world_size, shard_heads, hidden_dims, quant_scheme=None):
        if quant_scheme == "fp4":
            raise ValueError("prepost_backend='triton' does not support FP4 communication.")
        if quant_scheme == "fp8":
            payload, scale, _, _ = attn_pre_fp8(output, local_len, world_size, shard_heads, hidden_dims)
            return ((payload, scale),)
        return ((attn_pre(output, local_len, world_size, shard_heads, hidden_dims), None),)

    @staticmethod
    def unpack_attn(packed, output_dtype, hidden_dims):
        payload, scale = packed[0]
        if scale is None:
            return attn_post(payload)
        return attn_post_fp8(payload, scale, payload.shape, scale.shape, output_dtype)


def create_ulysses_prepost_backend(name):
    if name == "torch":
        return TorchUlyssesPrePost()
    if name == "triton":
        return TritonUlyssesPrePost()
    raise ValueError(f"Unknown prepost_backend={name!r}; expected 'torch' or 'triton'.")
