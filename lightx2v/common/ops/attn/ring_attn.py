import torch
import torch.distributed as dist

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate
from .utils.ring_comm import RingComm
from .utils.seq_p import (
    flatten_seq_p_tensor,
    pack_seq_p_tensor,
    split_main_aux_output,
    unpack_seq_p_tensor,
    validate_quant_scheme,
    validate_seq_p_inputs,
)


def _merge_attention_blocks(out, lse, block_out, block_lse):
    """Merge two attention partials represented by output and log-sum-exp."""
    if out is None:
        return block_out, block_lse
    new_lse = torch.logaddexp(lse, block_lse)
    out = torch.exp(lse - new_lse).unsqueeze(-1) * out + torch.exp(block_lse - new_lse).unsqueeze(-1) * block_out
    return out, new_lse


@ATTN_WEIGHT_REGISTER("ring")
class RingAttnWeight(AttnWeightTemplate):
    """Dense Ring attention over a sharded main sequence.

    The new interface supports one local main sequence and optional replicated
    auxiliary Q/K/V. Packed-varlen and general cross-attention layouts are
    rejected until they have independent correctness tests. ``head_parallel``
    is rejected because the Ring K/V rotation already overlaps communication
    with attention computation; splitting the work by head is not part of its
    execution model.
    """

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
        seq_p_prepost_backend="torch",
        seq_p_a2a_backend="torch",
        seq_p_quant_scheme=None,
        img_first=True,
        q_only_img=False,
        aux_first=False,
        **kwargs,
    ):
        """Deprecated compatibility adapter for the legacy joined-token API."""
        if not img_first or q_only_img or aux_first:
            raise ValueError("RingAttn legacy apply only supports image-only self-attention with img_first=True and no auxiliary region.")
        q = flatten_seq_p_tensor(q, "q")
        k = flatten_seq_p_tensor(k, "k")
        v = flatten_seq_p_tensor(v, "v")
        self._legacy_validate_layout(q, slice_qkv_len, cu_seqlens_qkv)
        if use_fp8_comm and use_fp4_comm:
            raise ValueError("use_fp8_comm and use_fp4_comm cannot both be enabled.")
        legacy_quant_scheme = "fp8" if use_fp8_comm else "fp4" if use_fp4_comm else None
        if seq_p_quant_scheme is not None and legacy_quant_scheme is not None and seq_p_quant_scheme != legacy_quant_scheme:
            raise ValueError("seq_p_quant_scheme conflicts with legacy communication flags.")
        quant_scheme = seq_p_quant_scheme if seq_p_quant_scheme is not None else legacy_quant_scheme
        output, aux_output = self.apply_new(
            q=q,
            k=k,
            v=v,
            attention_module=attention_module,
            seq_p_group=seq_p_group,
            prepost_backend=seq_p_prepost_backend,
            a2a_backend=seq_p_a2a_backend,
            quant_scheme=quant_scheme,
            tensor_fusion=use_tensor_fusion,
            head_parallel=enable_head_parallel,
            attention_kwargs=kwargs,
        )
        if aux_output is not None:
            raise RuntimeError("RingAttn legacy adapter received an unexpected auxiliary output.")
        return output

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
        tensor_fusion=False,
        head_parallel=False,
        aux_first=False,
        attention_kwargs=None,
    ):
        """Run dense Ring attention over one local main sequence.

        Every rank must provide the same local Q/K/V sequence length; the caller
        is responsible for padding and partitioning before this hot-path interface.
        ``aux_*`` tensors are optional replicated auxiliary regions. Packed-varlen,
        causal, masked, and general cross-attention layouts are not supported.
        ``tensor_fusion`` packs K/V into one Ring communication payload.
        """
        q = flatten_seq_p_tensor(q, "q")
        k = flatten_seq_p_tensor(k, "k")
        v = flatten_seq_p_tensor(v, "v")
        aux_q = flatten_seq_p_tensor(aux_q, "aux_q")
        aux_k = flatten_seq_p_tensor(aux_k, "aux_k")
        aux_v = flatten_seq_p_tensor(aux_v, "aux_v")
        attention_kwargs = {} if attention_kwargs is None else attention_kwargs

        self._validate_inputs(q, k, v, aux_q, aux_k, aux_v)
        self._validate_options(prepost_backend, a2a_backend, quant_scheme, head_parallel)
        self._validate_attention_module(attention_module)
        self._validate_attention_options(attention_kwargs)
        if not dist.is_available() or not dist.is_initialized():
            raise ValueError("RingAttn requires torch.distributed to be initialized.")

        output = self._apply_ring_pipeline(
            q,
            k,
            v,
            seq_p_group,
            attention_module=attention_module,
            attention_kwargs=attention_kwargs,
            aux_q=aux_q,
            aux_k=aux_k,
            aux_v=aux_v,
            aux_first=aux_first,
            kv_fusion=tensor_fusion,
            quant_scheme=quant_scheme,
        )
        aux_len = 0 if aux_q is None else aux_q.shape[0]
        main_output, aux_output = split_main_aux_output(output, q.shape[0], aux_len, aux_first)
        main_output = main_output.reshape(main_output.shape[0], -1)
        if aux_output is not None:
            aux_output = aux_output.reshape(aux_output.shape[0], -1)
        return main_output, aux_output

    @staticmethod
    def _validate_options(prepost_backend, a2a_backend, quant_scheme, head_parallel):
        prepost_backend = "torch" if prepost_backend is None else prepost_backend
        a2a_backend = "torch" if a2a_backend is None else a2a_backend
        if prepost_backend != "torch":
            raise ValueError(f"RingAttn only supports prepost_backend='torch', got {prepost_backend!r}.")
        if a2a_backend != "torch":
            raise ValueError(f"RingAttn only supports a2a_backend='torch', got {a2a_backend!r}.")
        if head_parallel:
            raise ValueError("RingAttn does not support head_parallel=True; Ring already overlaps communication with attention.")
        validate_quant_scheme(quant_scheme)

    @staticmethod
    def _validate_inputs(q, k, v, aux_q, aux_k, aux_v):
        validate_seq_p_inputs(q, k, v, aux_q, aux_k, aux_v)
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("RingAttn requires q, k, and v to have the same shape.")

    @staticmethod
    def _validate_attention_module(attention_module):
        if attention_module is None:
            raise ValueError("RingAttn requires an attention_module with a dense Ring attention contract.")
        if not callable(getattr(attention_module, "apply_with_lse", None)):
            raise ValueError("RingAttn requires attention_module.apply_with_lse(); backends that do not return per-block LSE natively are unsupported.")

    @staticmethod
    def _validate_attention_options(attention_kwargs):
        option_checks = (
            ("causal", attention_kwargs.get("causal") not in (None, False)),
            ("attn_mask", attention_kwargs.get("attn_mask") is not None),
            ("drop_rate", attention_kwargs.get("drop_rate") not in (None, 0, 0.0)),
            ("dropout_p", attention_kwargs.get("dropout_p") not in (None, 0, 0.0)),
            ("softcap", attention_kwargs.get("softcap") not in (None, 0, 0.0)),
            ("alibi_slopes", attention_kwargs.get("alibi_slopes") is not None),
        )
        for name, enabled in option_checks:
            if enabled:
                raise ValueError(f"RingAttn does not support attention option {name!r}.")

        window_size = attention_kwargs.get("window_size")
        if window_size is not None and window_size != (-1, -1) and window_size != [-1, -1]:
            raise ValueError("RingAttn does not support attention option 'window_size'.")

        for name in ("cu_seqlens_q", "cu_seqlens_kv", "max_seqlen_q", "max_seqlen_kv"):
            if attention_kwargs.get(name) is not None:
                raise ValueError(f"RingAttn does not support packed-varlen metadata {name!r}.")

    @staticmethod
    def _legacy_validate_layout(q, slice_qkv_len, cu_seqlens_qkv):
        split_len = int(slice_qkv_len.item()) if isinstance(slice_qkv_len, torch.Tensor) else int(slice_qkv_len)
        if split_len != q.shape[0]:
            raise ValueError("RingAttn image-only self-attention requires slice_qkv_len == local sequence length.")
        if cu_seqlens_qkv is None or len(cu_seqlens_qkv) != 2:
            raise ValueError("RingAttn only supports one image sequence; cu_seqlens_qkv must be [0, local_seq_len].")
        cu_start = int(cu_seqlens_qkv[0].item()) if isinstance(cu_seqlens_qkv[0], torch.Tensor) else int(cu_seqlens_qkv[0])
        cu_end = int(cu_seqlens_qkv[1].item()) if isinstance(cu_seqlens_qkv[1], torch.Tensor) else int(cu_seqlens_qkv[1])
        if (cu_start, cu_end) != (0, q.shape[0]):
            raise ValueError("RingAttn only supports cu_seqlens_qkv == [0, local_seq_len].")

    @staticmethod
    def _normalize_attention_output(output, q, method_name):
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"RingAttn attention_module.{method_name}() must return a torch.Tensor output.")
        if output.device != q.device:
            raise ValueError(f"RingAttn attention_module.{method_name}() must return output on the same device as q, got output.device={output.device}, q.device={q.device}.")
        if not output.is_floating_point():
            raise ValueError(f"RingAttn attention_module.{method_name}() must return a floating-point output.")
        if output.ndim == 3 and output.shape == q.shape:
            return output
        if output.ndim == 2 and output.shape == (q.shape[0], q.shape[1] * q.shape[2]):
            return output.reshape_as(q)
        if output.ndim == 4 and output.shape == (1, *q.shape):
            return output.reshape_as(q)
        raise ValueError(
            f"RingAttn attention_module.{method_name}() must return [local_seq, heads, head_dim], "
            "[local_seq, heads * head_dim], or [1, local_seq, heads, head_dim], "
            f"got shape={tuple(output.shape)} for q shape={tuple(q.shape)}."
        )

    @classmethod
    def _apply_attention_block(cls, attention_module, q, k, v, attention_kwargs):
        result = attention_module.apply_with_lse(
            q,
            k,
            v,
            softmax_scale=attention_kwargs.get("softmax_scale"),
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("RingAttn attention_module.apply_with_lse() must return (output, lse).")
        output, lse = result
        output = cls._normalize_attention_output(output, q, "apply_with_lse")
        if not isinstance(lse, torch.Tensor):
            raise TypeError("RingAttn attention_module.apply_with_lse() must return LSE as a torch.Tensor.")
        if lse.device != q.device:
            raise ValueError(f"RingAttn attention_module.apply_with_lse() must return LSE on the same device as q, got lse.device={lse.device}, q.device={q.device}.")
        if not lse.is_floating_point():
            raise ValueError("RingAttn attention_module.apply_with_lse() must return floating-point LSE.")
        expected_lse_shape = (q.shape[0], q.shape[1])
        if lse.shape != expected_lse_shape:
            raise ValueError(f"RingAttn attention_module.apply_with_lse() must return LSE with shape [local_seq, heads]={expected_lse_shape}, got {tuple(lse.shape)}.")
        return output.float(), lse.float()

    @staticmethod
    def _enqueue_packed(comm, packed):
        next_packed = []
        for payload, scale in packed:
            next_payload = comm.enqueue_send_recv(payload)
            next_scale = None if scale is None else comm.enqueue_send_recv(scale)
            next_packed.append((next_payload, next_scale))
        return tuple(next_packed)

    @classmethod
    def _apply_ring_pipeline(
        cls,
        q,
        k,
        v,
        seq_p_group,
        attention_module,
        attention_kwargs,
        aux_q,
        aux_k,
        aux_v,
        aux_first,
        kv_fusion,
        quant_scheme,
    ):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        hidden_dims = q.shape[-1]
        attention_q = q
        if aux_q is not None:
            attention_q = torch.cat((aux_q, q), dim=0) if aux_first else torch.cat((q, aux_q), dim=0)

        # Pack each source block once. The packed payload and optional scale are
        # forwarded unchanged around the ring, regardless of quantization.
        if kv_fusion:
            current_packed = (pack_seq_p_tensor(torch.cat((k, v), dim=0), quant_scheme),)
        else:
            current_packed = (
                pack_seq_p_tensor(k, quant_scheme),
                pack_seq_p_tensor(v, quant_scheme),
            )

        comm = RingComm(seq_p_group)
        world_size = comm.world_size
        out = None
        lse = None
        for step in range(world_size):
            if kv_fusion:
                current_kv = unpack_seq_p_tensor(current_packed[0], q.dtype, hidden_dims)
                current_k, current_v = current_kv.chunk(2, dim=0)
            else:
                current_k = unpack_seq_p_tensor(current_packed[0], q.dtype, hidden_dims)
                current_v = unpack_seq_p_tensor(current_packed[1], q.dtype, hidden_dims)

            if step + 1 != world_size:
                next_packed = cls._enqueue_packed(comm, current_packed)
                comm.commit()

            block_k, block_v = current_k, current_v
            if step + 1 == world_size and aux_k is not None:
                block_k = torch.cat((current_k, aux_k), dim=0)
                block_v = torch.cat((current_v, aux_v), dim=0)
            block_out, block_lse = cls._apply_attention_block(
                attention_module,
                attention_q,
                block_k,
                block_v,
                attention_kwargs,
            )
            out, lse = _merge_attention_blocks(out, lse, block_out, block_lse)

            if step + 1 != world_size:
                comm.wait()
                current_packed = next_packed
        return out.to(dtype=q.dtype)
