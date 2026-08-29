import torch

from lightx2v.utils.quant_utils import dequant_fp8_vllm, quant_fp8_vllm

INT8_MAX = 127.0

try:
    from sageattn3_sparse import dequant_fp4 as dequant_fp4_sage3
    from sageattn3_sparse import quant_fp4 as quant_fp4_sage3
except ImportError:
    quant_fp4_sage3 = None
    dequant_fp4_sage3 = None


def flatten_seq_p_tensor(tensor, name):
    if tensor is None:
        return None
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"Sequence-parallel attention supports one logical sequence only; 4D {name} must have batch size 1, got shape={tuple(tensor.shape)}.")
        return tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])
    return tensor


def validate_seq_p_inputs(q, k, v, aux_q, aux_k, aux_v):
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q/k/v must be 3D [sequence, heads, head_dim] tensors.")
    if q.shape[0] <= 0:
        raise ValueError("q/k/v must contain at least one local sequence token.")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q/k/v must have the same sequence length.")
    if k.shape != v.shape:
        raise ValueError("k/v must have the same shape.")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q/k/v must have the same head_dim.")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q/k/v must be on the same device.")
    if not q.is_floating_point() or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q/k/v must have the same floating-point dtype.")

    if (aux_k is None) != (aux_v is None):
        raise ValueError("aux k/v must either both be tensors or both be None.")
    if aux_q is not None and aux_k is None:
        raise ValueError("aux q requires aux k/v.")
    for name, tensor, source in zip(("q", "k", "v"), (aux_q, aux_k, aux_v), (q, k, v)):
        if tensor is None:
            continue
        if tensor.ndim != 3 or tensor.shape[1:] != source.shape[1:]:
            raise ValueError(f"aux_{name} must match {name} head dimensions, got {tuple(tensor.shape)} vs {tuple(source.shape)}.")
        if tensor.device != source.device:
            raise ValueError(f"aux_{name} must be on the same device as {name}.")
        if not tensor.is_floating_point() or tensor.dtype != source.dtype:
            raise ValueError(f"aux_{name} must have the same floating-point dtype as {name}.")
    if aux_k is not None and aux_k.shape[0] != aux_v.shape[0]:
        raise ValueError("aux k/v must have the same sequence length.")
    if aux_q is not None and aux_q.shape[0] != aux_k.shape[0]:
        raise ValueError("aux q/k/v must have the same sequence length.")


def split_main_aux_output(output, main_len, aux_len, aux_first):
    if aux_len == 0:
        return output, None
    if aux_first:
        return output[aux_len:], output[:aux_len]
    return output[:main_len], output[main_len:]


def validate_quant_scheme(quant_scheme):
    if quant_scheme not in (None, "fp8", "fp4", "int8"):
        raise ValueError(f"Unknown quant_scheme={quant_scheme!r}; expected None, 'fp8', 'fp4', or 'int8'.")
    if quant_scheme != "fp4":
        return
    if quant_fp4_sage3 is None or dequant_fp4_sage3 is None:
        raise ImportError("sageattn3_sparse quant_fp4/dequant_fp4 is required for sequence-parallel FP4 communication.")


def quant_int8_per_row(tensor):
    """Symmetrically quantize the last dimension to INT8 with FP32 row scales."""
    if not tensor.is_floating_point():
        raise ValueError(f"INT8 communication quantization requires a floating-point tensor, got {tensor.dtype}.")

    shape = tensor.shape
    hidden_dims = shape[-1]
    rows = tensor.reshape(-1, hidden_dims).float()
    amax = rows.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / INT8_MAX, torch.ones_like(amax))

    scaled = rows / scale
    rounded = torch.round(scaled)
    payload = rounded.clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return payload.reshape(shape).contiguous(), scale.reshape(*shape[:-1], 1).contiguous()


def dequant_int8_per_row(payload, scale, output_dtype):
    """Dequantize an INT8 communication payload using FP32 row scales."""
    if payload.dtype != torch.int8:
        raise ValueError(f"INT8 communication payload must have dtype torch.int8, got {payload.dtype}.")
    if scale.dtype != torch.float32:
        raise ValueError(f"INT8 communication scale must have dtype torch.float32, got {scale.dtype}.")
    expected_scale_shape = (*payload.shape[:-1], 1)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(f"INT8 communication scale shape must be {expected_scale_shape}, got {tuple(scale.shape)}.")
    return (payload.float() * scale).to(output_dtype)


def pack_seq_p_tensor(tensor, quant_scheme):
    tensor = tensor.contiguous()
    validate_quant_scheme(quant_scheme)
    if quant_scheme is None:
        return tensor, None

    shape = tensor.shape
    hidden_dims = shape[-1]
    if quant_scheme == "fp8":
        payload, scale = quant_fp8_vllm(tensor.reshape(-1, hidden_dims))
        return payload.reshape(shape).contiguous(), scale.reshape(*shape[:-1], 1).contiguous()
    if quant_scheme == "int8":
        return quant_int8_per_row(tensor)

    if hidden_dims % 16 != 0:
        raise ValueError(f"Sequence-parallel FP4 communication requires hidden_dims divisible by 16, got {hidden_dims}.")
    payload, scale = quant_fp4_sage3(tensor.reshape(1, 1, -1, hidden_dims))
    return payload.reshape(*shape[:-1], hidden_dims // 2).contiguous(), scale.reshape(*shape[:-1], hidden_dims // 16).contiguous()


def unpack_seq_p_tensor(packed, output_dtype, hidden_dims):
    payload, scale = packed
    if scale is None:
        return payload
    if payload.dtype == torch.float8_e4m3fn:
        return dequant_fp8_vllm(payload, scale, output_dtype)
    if payload.dtype == torch.int8 and payload.shape[-1] == hidden_dims and tuple(scale.shape) == (*payload.shape[:-1], 1):
        return dequant_int8_per_row(payload, scale, output_dtype)
    output_shape = (*payload.shape[:-1], hidden_dims)
    return dequant_fp4_sage3(
        payload.reshape(1, 1, -1, hidden_dims // 2),
        scale.reshape(1, 1, -1, hidden_dims // 16),
    ).reshape(output_shape)
