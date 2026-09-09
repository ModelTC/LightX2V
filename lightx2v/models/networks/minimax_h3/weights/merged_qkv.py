import torch
import torch.nn.functional as F

from lightx2v.common.ops.mm.mm_weight import MMWeight
from lightx2v.common.ops.utils import build_lora_and_diff_names
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


@MM_WEIGHT_REGISTER("h3ref_sgl_merged_qkv")
class MiniMaxH3SGLMergedQKVWeight(MMWeight):
    """Bias-free BF16 QKV projection stored in SGL's packed ``[out, in]`` layout."""

    supports_block_offload = True

    def __init__(
        self,
        weight_names,
        bias_name=None,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="transformer_blocks",
        lora_path="",
    ):
        self.source_weight_names = tuple(weight_names)
        if len(self.source_weight_names) != 3:
            raise ValueError(f"MiniMax-H3 merged QKV expects three source weights, got {self.source_weight_names}")
        if bias_name is not None:
            raise ValueError("MiniMax-H3 merged QKV is bias-free")
        if create_cpu_buffer or lazy_load:
            raise NotImplementedError("MiniMax-H3 merged QKV does not support CPU buffers or disk lazy loading")

        q_weight_name = self.source_weight_names[0]
        if not q_weight_name.endswith(".to_q.weight"):
            raise ValueError(f"Unexpected MiniMax-H3 Q weight name {q_weight_name!r}")
        packed_weight_name = q_weight_name.removesuffix(".to_q.weight") + ".qkv_packed.weight"
        super().__init__(
            weight_name=packed_weight_name,
            bias_name=None,
            create_cuda_buffer=create_cuda_buffer,
            create_cpu_buffer=False,
            lazy_load=False,
            lazy_load_file=lazy_load_file,
            is_post_adapter=is_post_adapter,
            lora_prefix=lora_prefix,
            lora_path=lora_path,
        )

        self.base_attrs = [(self.weight_name, "weight", False)]
        self.weight_need_transpose = False
        self.weight = None
        self.pin_weight = None
        self.bias = None
        self.local_qkv_dim = None
        self._source_adapter_names = set()
        for source_name in self.source_weight_names:
            self._source_adapter_names.update(build_lora_and_diff_names(source_name, lora_prefix))

    def _pack_source_weights(self, weight_dict):
        missing = [name for name in self.source_weight_names if name not in weight_dict]
        if missing:
            raise KeyError(f"MiniMax-H3 merged QKV is missing source weights: {missing}")

        source_weights = [weight_dict[name] for name in self.source_weight_names]
        shapes = [tuple(weight.shape) for weight in source_weights]
        if any(weight.ndim != 2 for weight in source_weights):
            raise ValueError(f"MiniMax-H3 Q/K/V weights must be two-dimensional, got {shapes}")
        if len({weight.shape for weight in source_weights}) != 1:
            raise ValueError(f"MiniMax-H3 Q/K/V weights must have identical shapes, got {shapes}")
        if len({weight.dtype for weight in source_weights}) != 1 or source_weights[0].dtype is not torch.bfloat16:
            raise TypeError(f"MiniMax-H3 merged QKV requires BF16 source weights, got {[weight.dtype for weight in source_weights]}")
        if len({weight.device for weight in source_weights}) != 1:
            raise ValueError(f"MiniMax-H3 Q/K/V weights must be on one device, got {[weight.device for weight in source_weights]}")

        self.local_qkv_dim = source_weights[0].shape[0]
        return torch.cat(source_weights, dim=0).contiguous()

    def load(self, weight_dict):
        packed_weight = self._pack_source_weights(weight_dict)
        super().load({self.weight_name: packed_weight})
        if not self.create_cuda_buffer:
            for source_name in self.source_weight_names:
                weight_dict.pop(source_name)

    def apply(self, input_tensor):
        if input_tensor.dtype is not torch.bfloat16 or not input_tensor.is_cuda:
            raise TypeError(f"MiniMax-H3 merged QKV requires a CUDA BF16 activation, got device={input_tensor.device}, dtype={input_tensor.dtype}")
        weight = self._get_actual_weight()
        if weight is None:
            raise RuntimeError("MiniMax-H3 merged QKV weight is not resident; move its WeightModule to the execution device first")
        if weight.device != input_tensor.device:
            raise RuntimeError(f"MiniMax-H3 merged QKV weight is on {weight.device}, but its activation is on {input_tensor.device}")
        return F.linear(input_tensor, weight)

    def split_qkv(self, projected):
        if self.local_qkv_dim is None:
            raise RuntimeError("MiniMax-H3 merged QKV has not been loaded")
        expected_width = 3 * self.local_qkv_dim
        if projected.shape[-1] != expected_width:
            raise ValueError(f"MiniMax-H3 merged QKV output width must be {expected_width}, got {projected.shape[-1]}")
        return projected.split(self.local_qkv_dim, dim=-1)

    def _reject_source_adapters(self, weight_dict):
        present = sorted(self._source_adapter_names.intersection(weight_dict))
        if present:
            raise NotImplementedError(f"MiniMax-H3 merged QKV does not support LoRA or diff weights: {present[:3]}")

    def register_diff(self, weight_dict):
        self._reject_source_adapters(weight_dict)

    def register_lora(self, weight_dict, strength):
        self._reject_source_adapters(weight_dict)

    def update_lora(self, weight_dict, strength):
        self._reject_source_adapters(weight_dict)

    def remove_lora(self):
        pass

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        raise NotImplementedError("MiniMax-H3 merged QKV does not support disk lazy loading")


__all__ = ["MiniMaxH3SGLMergedQKVWeight"]
