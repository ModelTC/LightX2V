import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.common.ops.mm.mm_weight import MMWeight, MMWeightTemplate
from lightx2v.common.ops.utils import create_default_tensors
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER
from lightx2v_platform.ops.utils.biren_supa import convBB, convSB
from lightx2v_platform.registry_factory import PLATFORM_MM_WEIGHT_REGISTER


@MM_WEIGHT_REGISTER("biren_default_linear")
class MMWeightBirenDefaultLinear(MMWeight):
    """Biren default-dtype weight: stores [out, in] and uses F.linear."""

    def _get_base_attrs_mapping(self):
        self.base_attrs = [
            (self.weight_name, "weight", False),
        ]
        if self.bias_name is not None:
            self.base_attrs.append((self.bias_name, "bias", False))

    def apply(self, input_tensor):
        weight = self._get_actual_weight()
        bias = self._get_actual_bias()
        if not self.has_lora_branch:
            return F.linear(input_tensor, weight, bias)
        return F.linear(input_tensor, weight, bias) + self.apply_lora(input_tensor)


@MM_WEIGHT_REGISTER("biren_tp2die")
class MMWeightBirenTP2Die(MMWeightBirenDefaultLinear):
    """TP-2Die weight: stores [out, in] and uses F.linear."""


@MM_WEIGHT_REGISTER("biren_force_fp32")
class MMWeightBirenForceFp32(MMWeight):
    """fp32 weight that avoids torch.addmm(..., out=...)."""

    def load(self, weight_dict):
        if not self.create_cuda_buffer and not self.create_cpu_buffer and not self.lazy_load:
            device_tensors, pin_tensors = create_default_tensors(self.base_attrs, weight_dict)
            self.weight = device_tensors.get("weight")
            self.bias = device_tensors.get("bias")
            self.pin_weight = pin_tensors.get("weight")
            self.pin_bias = pin_tensors.get("bias")
            for attr in ("weight", "bias", "pin_weight", "pin_bias"):
                tensor = getattr(self, attr, None)
                if tensor is not None:
                    setattr(self, attr, tensor.to(torch.float32))

    def apply(self, input_tensor):
        weight = self._get_actual_weight()
        bias = self._get_actual_bias()
        if not self.has_lora_branch:
            return input_tensor @ weight + (bias if bias is not None else 0)
        return input_tensor @ weight + (bias if bias is not None else 0) + self.apply_lora(input_tensor)


@MM_WEIGHT_REGISTER("biren_linear")
class MMWeightBirenLinear(MMWeight):
    """Default-dtype weight that avoids torch.addmm(..., out=...)."""

    def apply(self, input_tensor):
        weight = self._get_actual_weight()
        bias = self._get_actual_bias()
        if not self.has_lora_branch:
            return F.linear(input_tensor, weight.t(), bias)
        return F.linear(input_tensor, weight.t(), bias) + self.apply_lora(input_tensor)


@PLATFORM_MM_WEIGHT_REGISTER("supa_tensor_parallel")
class SupaTensorParallelMMWeight(MMWeightTemplate):
    def __init__(
        self,
        weight_name,
        bias_name,
        mm_type="Default",
        tp_group=None,
        tp_rank=0,
        tp_size=1,
        split_dim="col",
        gather_output=False,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
        assert split_dim in ["col", "row"], f"split_dim must be 'col' or 'row', got {split_dim}"
        self.tp_group = tp_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.split_dim = split_dim
        self.gather_output = gather_output
        from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER

        self._mm = MM_WEIGHT_REGISTER.get(mm_type, MMWeight)(
            weight_name=weight_name,
            bias_name=bias_name,
            create_cuda_buffer=create_cuda_buffer,
            create_cpu_buffer=create_cpu_buffer,
            lazy_load=lazy_load,
            lazy_load_file=lazy_load_file,
            is_post_adapter=is_post_adapter,
            lora_prefix=lora_prefix,
            lora_path=lora_path,
        )
        self._row_split_bias = None

    def load(self, weight_dict):
        self._mm.load(weight_dict)
        if self.split_dim == "row" and self.bias_name is not None and self.bias_name in weight_dict:
            bias = getattr(self._mm, "bias", None)
            if bias is not None:
                self._row_split_bias = bias.clone()
                self._mm.bias = None
        self._apply_weight_sbp()

    def apply(self, input_tensor):
        output = self._mm.apply(input_tensor)
        if self.split_dim == "col" and self.gather_output:
            output = convBB(output)
            if self.tp_size > 1 and self.tp_group is not None:
                output = self._all_gather_output(output)
        if self.split_dim == "row" and self.tp_size > 1 and self.tp_group is not None:
            output = convBB(output)
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)
            if self._row_split_bias is not None:
                output = output + self._row_split_bias
        elif self.split_dim == "row" and self._row_split_bias is not None:
            output = output + self._row_split_bias
        return output

    def _all_gather_output(self, output):
        gathered = [torch.empty_like(output) for _ in range(self.tp_size)]
        dist.all_gather(gathered, output, group=self.tp_group)
        return torch.cat(gathered, dim=-1)

    def _apply_weight_sbp(self):
        inner = self._mm
        if getattr(inner, "weight_cuda_buffer", None) is not None:
            return
        weight = getattr(inner, "weight", None)
        if weight is not None and hasattr(weight, "device") and weight.device.type == "supa":
            if self.split_dim == "col":
                inner.weight = convSB(weight, axis=0, layout="colmajor")
            else:
                inner.weight = convSB(weight, axis=1, layout="colmajor")
        bias = getattr(inner, "bias", None)
        if self.split_dim == "col" and bias is not None and hasattr(bias, "device") and bias.device.type == "supa":
            inner.bias = convSB(bias, axis=0, layout="linear_bias")
        if self._row_split_bias is not None and hasattr(self._row_split_bias, "device") and self._row_split_bias.device.type == "supa":
            self._row_split_bias = convBB(self._row_split_bias, layout="linear_bias")

    def _has_pin_attr(self, attr):
        return getattr(self._mm, f"pin_{attr}", None) is not None

    def _offload_to_pinned_cpu_refs(self):
        has_pin = False
        for attr in ("weight", "bias"):
            if self._has_pin_attr(attr):
                setattr(self._mm, attr, None)
                has_pin = True
        if self.split_dim == "row" and self._has_pin_attr("bias"):
            self._row_split_bias = None
            has_pin = True
        return has_pin

    def _move_row_bias_after_reduce(self):
        if self.split_dim != "row":
            return
        bias = getattr(self._mm, "bias", None)
        if bias is not None:
            self._row_split_bias = bias
            self._mm.bias = None

    def state_dict(self, destination=None):
        return self._mm.state_dict(destination)

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        self._mm.load_state_dict(destination, block_index, adapter_block_index)
        if self.split_dim == "row" and self._mm.bias is not None:
            self._row_split_bias = self._mm.bias
            self._mm.bias = None
        self._apply_weight_sbp()

    def to_cuda(self, non_blocking=False):
        self._mm.to_cuda(non_blocking)
        self._move_row_bias_after_reduce()
        self._apply_weight_sbp()

    def to_cpu(self, non_blocking=False):
        if self._offload_to_pinned_cpu_refs():
            return
        self._mm.to_cpu(non_blocking)
        if self._row_split_bias is not None and getattr(self._row_split_bias, "device", None) is not None and self._row_split_bias.device.type != "cpu":
            self._row_split_bias = self._row_split_bias.to("cpu", non_blocking=non_blocking)
