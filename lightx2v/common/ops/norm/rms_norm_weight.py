import re
from abc import ABCMeta, abstractmethod

import torch

from lightx2v.utils.envs import *
from lightx2v.utils.registry_factory import RMS_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE
try:
    import sgl_kernel
except ImportError:
    sgl_kernel = None


class RMSWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name, create_cuda_buffer=False, create_cpu_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False, eps=1e-6):
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

    def load(self, weight_dict):
        if self.create_cuda_buffer:
            self._load_cuda_buffer(weight_dict)
        elif self.create_cpu_buffer:
            self._load_cpu_pin_buffer()
        else:
            self._load_default_tensors(weight_dict)

    def _load_default_tensors(self, weight_dict):
        if not self.lazy_load:
            device = weight_dict[self.weight_name].device
            if device.type == "cpu":
                weight_tensor = weight_dict[self.weight_name]
                self.pin_weight = self._create_cpu_pin_weight(weight_tensor)
                del weight_dict[self.weight_name]
            else:
                self.weight = weight_dict[self.weight_name]

    def _get_weight_tensor(self, weight_dict=None, use_infer_dtype=False):
        if self.lazy_load:
            tensor = self.lazy_load_file.get_tensor(self.weight_name)
            if use_infer_dtype:
                tensor = tensor.to(self.infer_dtype)
        else:
            tensor = weight_dict[self.weight_name]
        return tensor

    def _create_cpu_pin_weight(self, tensor):
        pin_tensor = torch.empty(tensor.shape, pin_memory=True, dtype=tensor.dtype)
        pin_tensor.copy_(tensor)
        del tensor
        return pin_tensor

    def _load_cuda_buffer(self, weight_dict):
        weight_tensor = self._get_weight_tensor(weight_dict, use_infer_dtype=self.lazy_load)
        self.weight_cuda_buffer = weight_tensor.to(AI_DEVICE)

    def _load_cpu_pin_buffer(self):
        weight_tensor = self._get_weight_tensor(use_infer_dtype=True)
        self.pin_weight = self._create_cpu_pin_weight(weight_tensor)
            
    @abstractmethod
    def apply(self, input_tensor):
        pass

    def set_config(self, config=None):
        if config is not None:
            self.config = config

    def to_cuda(self, non_blocking=False):
        self.weight = self.pin_weight.to(AI_DEVICE, non_blocking=non_blocking)

    def to_cpu(self, non_blocking=False):
        if hasattr(self, "pin_weight"):
            self.weight = self.pin_weight.copy_(self.weight, non_blocking=non_blocking).cpu()
        else:
            self.weight = self.weight.to("cpu", non_blocking=non_blocking)

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        destination[self.weight_name] = self.pin_weight if hasattr(self, "pin_weight") else self.weight
        return destination

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        if self.is_post_adapter:
            assert adapter_block_index is not None
            weight_name = re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", self.weight_name, count=1)
        else:
            weight_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.weight_name, count=1)

        if weight_name not in destination:
            self.weight = None
            return
        self.weight = self.weight_cuda_buffer.copy_(destination[weight_name], non_blocking=True)

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        if self.is_post_adapter:
            self.weight_name = re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", self.weight_name, count=1)
        else:
            self.weight_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.weight_name, count=1)

        weight_tensor = self.lazy_load_file.get_tensor(self.weight_name).to(self.infer_dtype)
        self.pin_weight = self.pin_weight.copy_(weight_tensor)
        del weight_tensor


@RMS_WEIGHT_REGISTER("Default")
class RMSWeight(RMSWeightTemplate):
    def __init__(self, weight_name, create_cuda_buffer=False, create_cpu_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False, eps=1e-6):
        super().__init__(weight_name, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file, is_post_adapter, eps)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def apply(self, input_tensor):
        if GET_SENSITIVE_DTYPE() != GET_DTYPE():
            input_tensor = self._norm(input_tensor).type_as(input_tensor) * self.weight
        else:
            input_tensor = self._norm(input_tensor.float()).type_as(input_tensor) * self.weight
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
    ):
        super().__init__(weight_name, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file, is_post_adapter, eps)

    def apply(self, input_tensor):
        if sgl_kernel is not None and self.sensitive_layer_dtype == self.infer_dtype:
            input_tensor = input_tensor.contiguous()
            orig_shape = input_tensor.shape
            input_tensor = input_tensor.view(-1, orig_shape[-1])
            input_tensor = sgl_kernel.rmsnorm(input_tensor, self.weight, self.eps).view(orig_shape)
        else:
            # sgl_kernel is not available or dtype!=torch.bfloat16/float16, fallback to default implementation
            if self.sensitive_layer_dtype != self.infer_dtype:
                input_tensor = input_tensor * torch.rsqrt(input_tensor.float().pow(2).mean(-1, keepdim=True) + self.eps).to(self.infer_dtype)
                input_tensor = (input_tensor * self.weight).to(self.infer_dtype)
            else:
                input_tensor = input_tensor * torch.rsqrt(input_tensor.pow(2).mean(-1, keepdim=True) + self.eps)
                input_tensor = input_tensor * self.weight

        return input_tensor


@RMS_WEIGHT_REGISTER("fp32_variance")
class RMSWeightFP32(RMSWeight):
    def __init__(self, weight_name, create_cuda_buffer=False, create_cpu_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False, eps=1e-6):
        super().__init__(weight_name, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file, is_post_adapter, eps)

    def apply(self, input_tensor):
        input_dtype = input_tensor.dtype
        variance = input_tensor.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = input_tensor * torch.rsqrt(variance + self.eps)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight
        hidden_states = hidden_states.to(input_dtype)

        return hidden_states


@RMS_WEIGHT_REGISTER("self_forcing")
class RMSWeightSF(RMSWeight):
    def __init__(self, weight_name, create_cuda_buffer=False, create_cpu_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False, eps=1e-6):
        super().__init__(weight_name, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file, is_post_adapter, eps)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def apply(self, x):
        return self._norm(x.float()).type_as(x) * self.weight