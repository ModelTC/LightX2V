from abc import ABCMeta, abstractmethod

import torch

from lightx2v.utils.envs import *
from lightx2v.utils.registry_factory import RMS_WEIGHT_REGISTER

try:
    import sgl_kernel
except ImportError:
    sgl_kernel = None


class RMSWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name, lazy_load=False, lazy_load_file=None, eps=1e-6):
        self.weight_name = weight_name
        self.eps = eps
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.config = {}

    def load(self, weight_dict):
        if not self.lazy_load:
            self.weight = weight_dict[self.weight_name]
            self.pinned_weight = torch.empty(self.weight.shape, pin_memory=True, dtype=self.weight.dtype)

    def clear(self):
        attrs = ["weight", "pinned_weight"]
        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)
                setattr(self, attr, None)

    @abstractmethod
    def apply(self, input_tensor):
        pass

    def set_config(self, config=None):
        if config is not None:
            self.config = config

    def to_cpu(self, non_blocking=False):
        if hasattr(self, "pinned_weight"):
            self.weight = self.pinned_weight.copy_(self.weight, non_blocking=non_blocking).cpu()
        else:
            self.weight = self.weight.to("cpu", non_blocking=non_blocking)

    def to_cuda(self, non_blocking=False):
        self.weight = self.weight.cuda(non_blocking=non_blocking)

    def _calculate_size(self):
        return self.weight.numel() * self.weight.element_size()


@RMS_WEIGHT_REGISTER("Default")
class RMSWeight(RMSWeightTemplate):
    def __init__(self, weight_name, lazy_load=False, lazy_load_file=None, eps=1e-6):
        super().__init__(weight_name, lazy_load, lazy_load_file, eps)

    def load(self, weight_dict):
        if not self.lazy_load:
            self.weight = weight_dict[self.weight_name]
            self.pinned_weight = torch.empty(self.weight.shape, pin_memory=True, dtype=self.weight.dtype)

    def load_from_disk(self):
        if not torch._dynamo.is_compiling():
            self.weight = self.lazy_load_file.get_tensor(self.weight_name).to(torch.bfloat16).pin_memory()
        else:
            self.weight = self.lazy_load_file.get_tensor(self.weight_name).to(torch.bfloat16)

    def apply(self, input_tensor):
        if GET_DTYPE() == "BF16":
            input_tensor = input_tensor * torch.rsqrt(input_tensor.pow(2).mean(-1, keepdim=True) + self.eps)
            input_tensor = input_tensor * self.weight
        else:
            input_tensor = input_tensor * torch.rsqrt(input_tensor.float().pow(2).mean(-1, keepdim=True) + self.eps)
            input_tensor = (input_tensor * self.weight).to(torch.bfloat16)

        return input_tensor

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        destination[self.weight_name] = self.weight.cpu().detach().clone()
        return destination


@RMS_WEIGHT_REGISTER("sgl-kernel")
class RMSWeightSgl(RMSWeight):
    def __init__(self, weight_name, lazy_load=False, lazy_load_file=None, eps=1e-6):
        super().__init__(weight_name, lazy_load, lazy_load_file, eps)

    def load(self, weight_dict):
        if not self.lazy_load:
            self.weight = weight_dict[self.weight_name]
            self.pinned_weight = torch.empty(self.weight.shape, pin_memory=True, dtype=self.weight.dtype)

    def load_from_disk(self):
        if not torch._dynamo.is_compiling():
            self.weight = self.lazy_load_file.get_tensor(self.weight_name).to(torch.bfloat16).pin_memory()
        else:
            self.weight = self.lazy_load_file.get_tensor(self.weight_name).to(torch.bfloat16)

    def apply(self, input_tensor):
        use_bf16 = GET_DTYPE() == "BF16"
        if sgl_kernel is not None and use_bf16:
            input_tensor = input_tensor.contiguous()
            orig_shape = input_tensor.shape
            input_tensor = input_tensor.view(-1, orig_shape[-1])
            input_tensor = sgl_kernel.rmsnorm(input_tensor, self.weight, self.eps).view(orig_shape)
        else:
            # sgl_kernel is not available or dtype!=torch.bfloat16, fallback to default implementation
            if use_bf16:
                input_tensor = input_tensor * torch.rsqrt(input_tensor.pow(2).mean(-1, keepdim=True) + self.eps)
                input_tensor = input_tensor * self.weight
            else:
                input_tensor = input_tensor * torch.rsqrt(input_tensor.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(input_tensor)
                input_tensor = (input_tensor * self.weight).type_as(input_tensor)

        return input_tensor
