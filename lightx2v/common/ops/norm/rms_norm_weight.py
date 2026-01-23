from abc import ABCMeta, abstractmethod

import torch
import torch.distributed as dist
from loguru import logger
from safetensors import safe_open

from lightx2v.common.ops.norm.triton_ops import rms_norm_kernel
from lightx2v.common.ops.utils import *
from lightx2v.utils.envs import *
from lightx2v.utils.registry_factory import RMS_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

try:
    import sgl_kernel
except ImportError:
    sgl_kernel = None


class RMSWeightTemplate(metaclass=ABCMeta):
    def __init__(
        self,
        weight_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        eps=1e-6,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        self.weight_name = weight_name
        self.eps = eps
        self.create_cuda_buffer = create_cuda_buffer
        self.create_cpu_buffer = create_cpu_buffer
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.is_post_adapter = is_post_adapter
        self.infer_dtype = GET_DTYPE()
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()
        self.config = {}
        self.lora_prefix = lora_prefix
        self.lora_path = lora_path
        self.has_lora_branch = False
        self.has_diff = False
        self._get_base_attrs_mapping()
        self._get_lora_attr_mapping()

    def _get_base_attrs_mapping(self):
        self.base_attrs = []
        self.base_attrs.append((self.weight_name, "weight", False))

    def _get_lora_attr_mapping(self):
        _, _, _, self.weight_diff_name, _ = build_lora_and_diff_names(self.weight_name, self.lora_prefix)
        self.lora_attrs = {
            "weight_diff": "weight_diff_name",
        }
        self.weight_diff = torch.tensor(0.0, dtype=GET_DTYPE(), device=AI_DEVICE)

    def _get_actual_weight(self):
        if not hasattr(self, "weight_diff"):
            return self.weight
        return self.weight + self.weight_diff

    def register_diff(self, weight_dict):
        if not self.lazy_load or self.create_cuda_buffer or self.create_cpu_buffer:
            if self.weight_diff_name in weight_dict:
                self.weight_diff = weight_dict[self.weight_diff_name]
                logger.debug(f"Register Diff to {self.weight_name}")

    def load(self, weight_dict):
        if not self.create_cuda_buffer and not self.create_cpu_buffer and not self.lazy_load:
            device_tensors, pin_tensors = create_default_tensors(self.base_attrs, weight_dict)
            self.weight = device_tensors.get("weight")
            self.pin_weight = pin_tensors.get("weight")
        elif self.create_cuda_buffer:
            result = create_cuda_buffers(
                self.base_attrs,
                weight_dict,
                self.lazy_load,
                self.lazy_load_file,
                use_infer_dtype=True,
            )
            self.weight_cuda_buffer = result.get("weight")
        elif self.create_cpu_buffer:
            result = create_cpu_buffers(self.base_attrs, self.lazy_load_file, use_infer_dtype=True)
            self.pin_weight = result.get("weight")
            self.weight = None

    def set_config(self, config=None):
        if config is not None:
            self.config = config

    def to_cuda(self, non_blocking=False):
        move_attr_to_cuda(self, self.base_attrs, self.lora_attrs, non_blocking)

    def to_cpu(self, non_blocking=False):
        move_attr_to_cpu(self, self.base_attrs, self.lora_attrs, non_blocking)

    def state_dict(self, destination=None):
        return state_dict(self, self.base_attrs, self.lora_attrs, destination)

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        return load_state_dict(
            self,
            self.base_attrs,
            self.lora_attrs,
            destination,
            block_index,
            adapter_block_index,
        )

    def load_lora_state_dict_from_disk(self, block_index):
        self.weight_diff_name = resolve_block_name(self.weight_diff_name, block_index)
        with safe_open(self.lora_path, framework="pt", device="cpu") as lora_load_file:
            for lora_attr, lora_attr_name in self.lora_attrs.items():
                if getattr(self, lora_attr_name) in lora_load_file.keys():
                    setattr(
                        self,
                        lora_attr,
                        getattr(self, lora_attr).copy_(
                            lora_load_file.get_tensor(getattr(self, lora_attr_name)),
                            non_blocking=True,
                        ),
                    )

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        if self.has_lora_branch or self.has_diff:
            self.load_lora_state_dict_from_disk(block_index)
        self.weight_name = resolve_block_name(self.weight_name, block_index, adapter_block_index, self.is_post_adapter)
        lazy_load_file_path = get_lazy_load_file_path(self.lazy_load_file, self.weight_name)
        with safe_open(lazy_load_file_path, framework="pt", device="cpu") as lazy_load_file:
            weight_tensor = lazy_load_file.get_tensor(self.weight_name).to(self.infer_dtype)
            self.pin_weight = self.pin_weight.copy_(weight_tensor)
        del weight_tensor

    @abstractmethod
    def apply(self, input_tensor):
        pass


@RMS_WEIGHT_REGISTER("Default")
class RMSWeight(RMSWeightTemplate):
    def __init__(
        self,
        weight_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        eps=1e-6,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            eps,
            lora_prefix,
            lora_path,
        )

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def apply(self, input_tensor):
        if GET_SENSITIVE_DTYPE() != GET_DTYPE():
            input_tensor = self._norm(input_tensor).type_as(input_tensor) * (self._get_actual_weight())
        else:
            input_tensor = self._norm(input_tensor.float()).type_as(input_tensor) * (self._get_actual_weight())
        return input_tensor


@RMS_WEIGHT_REGISTER("TensorParallel")
class RMSWeightTP(RMSWeightTemplate):
    """
    RMSNorm weight module with tensor parallelism support.

    The weight is split along the hidden dimension to match the split QKV outputs.
    """

    def __init__(
        self,
        weight_name,
        tp_group=None,
        tp_rank=0,
        tp_size=1,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        eps=1e-6,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            eps,
            lora_prefix,
            lora_path,
        )
        self.tp_group = tp_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size

    def apply(self, input_tensor):
        local_sum = input_tensor.pow(2).sum(-1, keepdim=True)

        # All-reduce to get global sum
        if self.tp_size > 1 and self.tp_group is not None:
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=self.tp_group)

        # Compute global mean: global_sum / hidden_dim
        hidden_dim = input_tensor.shape[-1] * self.tp_size
        global_mean = local_sum / hidden_dim

        # Apply normalization with global mean
        if self.sensitive_layer_dtype != self.infer_dtype:
            input_tensor = input_tensor * torch.rsqrt(global_mean.float() + self.eps).to(self.infer_dtype)
            input_tensor = (input_tensor * self._get_actual_weight()).to(self.infer_dtype)
        else:
            input_tensor = input_tensor * torch.rsqrt(global_mean + self.eps)
            input_tensor = input_tensor * self._get_actual_weight()
        return input_tensor


@RMS_WEIGHT_REGISTER("sgl-kernel")
class RMSWeightSgl(RMSWeight):
    def __init__(
        self,
        weight_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        eps=1e-6,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            eps,
            lora_prefix,
            lora_path,
        )

    def apply(self, input_tensor):
        if sgl_kernel is not None and self.sensitive_layer_dtype == self.infer_dtype:
            input_tensor = input_tensor.contiguous()
            orig_shape = input_tensor.shape
            input_tensor = input_tensor.view(-1, orig_shape[-1])
            input_tensor = sgl_kernel.rmsnorm(input_tensor, (self._get_actual_weight()), self.eps).view(orig_shape)
        else:
            # sgl_kernel is not available or dtype!=torch.bfloat16/float16, fallback to default implementation
            if self.sensitive_layer_dtype != self.infer_dtype:
                input_tensor = input_tensor * torch.rsqrt(input_tensor.float().pow(2).mean(-1, keepdim=True) + self.eps).to(self.infer_dtype)
                input_tensor = (input_tensor * (self._get_actual_weight())).to(self.infer_dtype)
            else:
                input_tensor = input_tensor * torch.rsqrt(input_tensor.pow(2).mean(-1, keepdim=True) + self.eps)
                input_tensor = input_tensor * (self._get_actual_weight())

        return input_tensor


@RMS_WEIGHT_REGISTER("fp32_variance")
class RMSWeightFP32(RMSWeight):
    def __init__(
        self,
        weight_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        eps=1e-6,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            eps,
            lora_prefix,
            lora_path,
        )

    def apply(self, input_tensor):
        input_dtype = input_tensor.dtype
        variance = input_tensor.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = input_tensor * torch.rsqrt(variance + self.eps)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        if self.weight is not None:
            hidden_states = hidden_states * (self._get_actual_weight())
        hidden_states = hidden_states.to(input_dtype)

        return hidden_states


@RMS_WEIGHT_REGISTER("self_forcing")
class RMSWeightSF(RMSWeight):
    def __init__(
        self,
        weight_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        eps=1e-6,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            eps,
            lora_prefix,
            lora_path,
        )

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def apply(self, x):
        return self._norm(x.float()).type_as(x) * (self._get_actual_weight())


@RMS_WEIGHT_REGISTER("one-pass")
class RMSWeightOnePass(RMSWeight):
    def __init__(
        self,
        weight_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        eps=1e-6,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            eps,
            lora_prefix,
            lora_path,
        )

    def apply(self, input_tensor):
        return rms_norm_kernel(input_tensor, (self._get_actual_weight()), self.eps)
