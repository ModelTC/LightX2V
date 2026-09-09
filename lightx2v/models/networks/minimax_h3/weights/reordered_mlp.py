import torch
import torch.nn.functional as F

from lightx2v.common.ops.mm.mm_weight import MMWeight
from lightx2v.common.ops.utils import build_lora_and_diff_names
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


@MM_WEIGHT_REGISTER("h3ref_sgl_reordered_mlp")
class MiniMaxH3SGLReorderedMLPWeight(MMWeight):
    """H3 SwiGLU input projection stored as contiguous ``[gate; value]`` rows."""

    supports_block_offload = True

    def __init__(
        self,
        weight_name,
        bias_name=None,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="transformer_blocks",
        lora_path="",
    ):
        if bias_name is not None:
            raise ValueError("MiniMax-H3 SwiGLU input projection is bias-free")
        if create_cpu_buffer or lazy_load:
            raise NotImplementedError("MiniMax-H3 reordered SwiGLU does not support CPU buffers or disk lazy loading")
        if not weight_name.endswith(".weight"):
            raise ValueError(f"Unexpected MiniMax-H3 SwiGLU weight name {weight_name!r}")

        self.source_weight_name = weight_name
        reordered_weight_name = weight_name.removesuffix(".weight") + ".reordered.weight"

        super().__init__(
            weight_name=reordered_weight_name,
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
        self._source_adapter_names = set(build_lora_and_diff_names(self.source_weight_name, lora_prefix))

    def _reorder_source_weight(self, weight):
        if weight.ndim != 2 or weight.shape[0] % 2:
            raise ValueError(f"Invalid MiniMax-H3 fused SwiGLU weight shape {tuple(weight.shape)}")
        if weight.dtype is not torch.bfloat16:
            raise TypeError(f"MiniMax-H3 reordered SwiGLU requires a BF16 source weight, got {weight.dtype}")
        value_weight, gate_weight = weight.chunk(2, dim=0)
        return torch.cat((gate_weight, value_weight), dim=0).contiguous()

    def load(self, weight_dict):
        if self.source_weight_name not in weight_dict:
            raise KeyError(f"MiniMax-H3 reordered SwiGLU is missing {self.source_weight_name}")
        reordered_weight = self._reorder_source_weight(weight_dict[self.source_weight_name])
        super().load({self.weight_name: reordered_weight})
        if not self.create_cuda_buffer:
            weight_dict.pop(self.source_weight_name)

    def apply(self, input_tensor):
        if input_tensor.dtype is not torch.bfloat16 or not input_tensor.is_cuda:
            raise TypeError(f"MiniMax-H3 reordered SwiGLU requires a CUDA BF16 activation, got device={input_tensor.device}, dtype={input_tensor.dtype}")
        weight = self._get_actual_weight()
        if weight is None:
            raise RuntimeError("MiniMax-H3 reordered SwiGLU weight is not resident; move its WeightModule to the execution device first")
        if weight.device != input_tensor.device:
            raise RuntimeError(f"MiniMax-H3 reordered SwiGLU weight is on {weight.device}, but its activation is on {input_tensor.device}")
        return F.linear(input_tensor, weight)

    def _reject_source_adapters(self, weight_dict):
        present = sorted(self._source_adapter_names.intersection(weight_dict))
        if present:
            raise NotImplementedError(f"MiniMax-H3 reordered SwiGLU does not support LoRA or diff weights: {present[:3]}")

    def register_diff(self, weight_dict):
        self._reject_source_adapters(weight_dict)

    def register_lora(self, weight_dict, strength):
        self._reject_source_adapters(weight_dict)

    def update_lora(self, weight_dict, strength):
        self._reject_source_adapters(weight_dict)

    def remove_lora(self):
        pass

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        raise NotImplementedError("MiniMax-H3 reordered SwiGLU does not support disk lazy loading")


__all__ = ["MiniMaxH3SGLReorderedMLPWeight"]
