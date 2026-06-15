import torch

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.common.ops.attn.flash_attn import flash_attn_varlen_func_v2, flash_attn_varlen_func_v3
from lightx2v.models.networks.wan.weights.dreamzero.pre_weights import DreamZeroCategoryLinearWeights
from lightx2v.models.networks.wan.weights.transformer_weights import WanTransformerWeights
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


@ATTN_WEIGHT_REGISTER("dreamzero_fa2")
class DreamZeroFlashAttn2Weight:
    def __init__(self):
        self.config = {}

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
        if flash_attn_varlen_func_v2 is None:
            raise ImportError("dreamzero_fa2 requires flash_attn_varlen_func from flash-attn 2.")

        causal = kwargs.get("causal", False)
        softmax_scale = kwargs.get("softmax_scale", None)
        dropout_p = kwargs.get("dropout_p", 0.0)
        window_size = kwargs.get("window_size", (-1, -1))

        if q.dim() == 4:
            batch_size, q_len = q.shape[:2]
            kv_len = k.shape[1]
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])
            if cu_seqlens_q is None:
                cu_seqlens_q = torch.arange(0, (batch_size + 1) * q_len, q_len, device=q.device, dtype=torch.int32)
            if cu_seqlens_kv is None:
                cu_seqlens_kv = torch.arange(0, (batch_size + 1) * kv_len, kv_len, device=k.device, dtype=torch.int32)
            max_seqlen_q = max_seqlen_q or q_len
            max_seqlen_kv = max_seqlen_kv or kv_len
        else:
            if cu_seqlens_q is None:
                cu_seqlens_q = torch.tensor([0, q.shape[0]], device=q.device, dtype=torch.int32)
            if cu_seqlens_kv is None:
                cu_seqlens_kv = torch.tensor([0, k.shape[0]], device=k.device, dtype=torch.int32)
            max_seqlen_q = max_seqlen_q or q.shape[0]
            max_seqlen_kv = max_seqlen_kv or k.shape[0]

        if cu_seqlens_q.is_cpu:
            cu_seqlens_q = cu_seqlens_q.to(q.device, non_blocking=True)
        if cu_seqlens_kv.is_cpu:
            cu_seqlens_kv = cu_seqlens_kv.to(k.device, non_blocking=True)

        out = flash_attn_varlen_func_v2(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )
        return out.reshape(q.shape[0], -1)


@ATTN_WEIGHT_REGISTER("dreamzero_cross_fa")
class DreamZeroCrossFlashAttnWeight(DreamZeroFlashAttn2Weight):
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
        causal = kwargs.get("causal", False)
        softmax_scale = kwargs.get("softmax_scale", None)
        dropout_p = kwargs.get("dropout_p", 0.0)
        window_size = kwargs.get("window_size", (-1, -1))
        deterministic = kwargs.get("deterministic", False)

        if q.dim() == 4:
            batch_size, q_len = q.shape[:2]
            kv_len = k.shape[1]
            q = q.reshape(-1, q.shape[-2], q.shape[-1])
            k = k.reshape(-1, k.shape[-2], k.shape[-1])
            v = v.reshape(-1, v.shape[-2], v.shape[-1])
            if cu_seqlens_q is None:
                cu_seqlens_q = torch.arange(0, (batch_size + 1) * q_len, q_len, device=q.device, dtype=torch.int32)
            if cu_seqlens_kv is None:
                cu_seqlens_kv = torch.arange(0, (batch_size + 1) * kv_len, kv_len, device=k.device, dtype=torch.int32)
            max_seqlen_q = max_seqlen_q or q_len
            max_seqlen_kv = max_seqlen_kv or kv_len
        else:
            if cu_seqlens_q is None:
                cu_seqlens_q = torch.tensor([0, q.shape[0]], device=q.device, dtype=torch.int32)
            if cu_seqlens_kv is None:
                cu_seqlens_kv = torch.tensor([0, k.shape[0]], device=k.device, dtype=torch.int32)
            max_seqlen_q = max_seqlen_q or q.shape[0]
            max_seqlen_kv = max_seqlen_kv or k.shape[0]

        if cu_seqlens_q.is_cpu:
            cu_seqlens_q = cu_seqlens_q.to(q.device, non_blocking=True)
        if cu_seqlens_kv.is_cpu:
            cu_seqlens_kv = cu_seqlens_kv.to(k.device, non_blocking=True)

        if flash_attn_varlen_func_v3 is not None:
            out = flash_attn_varlen_func_v3(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_kv,
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_kv,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic,
            )
            if isinstance(out, tuple):
                out = out[0]
            return out.reshape(q.shape[0], -1)

        if flash_attn_varlen_func_v2 is None:
            raise ImportError("dreamzero_cross_fa requires flash_attn_varlen_func from flash-attn 2 or 3.")
        out = flash_attn_varlen_func_v2(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )
        return out.reshape(q.shape[0], -1)


class DreamZeroActionDecoderWeights(WeightModule):
    def __init__(self):
        super().__init__()
        self.add_module("layer1", DreamZeroCategoryLinearWeights("action_decoder.layer1"))
        self.add_module("layer2", DreamZeroCategoryLinearWeights("action_decoder.layer2"))


class DreamZeroTransformerWeights(WanTransformerWeights):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__(config, lazy_load_path=lazy_load_path, lora_path=lora_path)
        for block in self.blocks:
            self._ensure_image_cross_attn(block.compute_phases[1])
        if getattr(self, "offload_block_cuda_buffers", None) is not None:
            for block in self.offload_block_cuda_buffers:
                self._ensure_image_cross_attn(block.compute_phases[1])
        if getattr(self, "offload_block_cpu_buffers", None) is not None:
            for block in self.offload_block_cpu_buffers:
                self._ensure_image_cross_attn(block.compute_phases[1])
        if getattr(self, "offload_phase_cuda_buffers", None) is not None:
            self._ensure_image_cross_attn(self.offload_phase_cuda_buffers[1])
        if getattr(self, "offload_phase_cpu_buffers", None) is not None:
            for phases in self.offload_phase_cpu_buffers:
                self._ensure_image_cross_attn(phases[1])
        self.add_module("action_decoder", DreamZeroActionDecoderWeights())

    def _ensure_image_cross_attn(self, cross_attn):
        if hasattr(cross_attn, "cross_attn_k_img"):
            return
        block_prefix = "blocks"
        block_index = cross_attn.block_index
        create_cuda_buffer = getattr(cross_attn, "create_cuda_buffer", False)
        create_cpu_buffer = getattr(cross_attn, "create_cpu_buffer", False)
        lazy_load = cross_attn.lazy_load
        lazy_load_file = cross_attn.lazy_load_file
        lora_path = None
        cross_attn.add_module(
            "cross_attn_k_img",
            MM_WEIGHT_REGISTER[cross_attn.mm_type](
                f"{block_prefix}.{block_index}.cross_attn.k_img.weight",
                f"{block_prefix}.{block_index}.cross_attn.k_img.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )
        cross_attn.add_module(
            "cross_attn_v_img",
            MM_WEIGHT_REGISTER[cross_attn.mm_type](
                f"{block_prefix}.{block_index}.cross_attn.v_img.weight",
                f"{block_prefix}.{block_index}.cross_attn.v_img.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )
        cross_attn.add_module(
            "cross_attn_norm_k_img",
            RMS_WEIGHT_REGISTER[cross_attn.attn_rms_norm_type](
                f"{block_prefix}.{block_index}.cross_attn.norm_k_img.weight",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )
        cross_attn.add_module("cross_attn_2", ATTN_WEIGHT_REGISTER[self.config["cross_attn_2_type"]]())

    def non_block_weights_to_cuda(self):
        super().non_block_weights_to_cuda()
        self.action_decoder.to_cuda()

    def non_block_weights_to_cpu(self):
        super().non_block_weights_to_cpu()
        self.action_decoder.to_cpu()
