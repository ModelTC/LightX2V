import re
from abc import ABCMeta, abstractmethod

import torch

from lightx2v.utils.envs import *
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER

from .triton_ops import norm_infer


class LNWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name=None, bias_name=None, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False, eps=1e-6):
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.eps = eps
        self.create_cuda_buffer = create_cuda_buffer
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.is_post_adapter = is_post_adapter
        self.config = {}
        self.infer_dtype = GET_DTYPE()
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()

    def load(self, weight_dict):
        if not self.lazy_load:
            if self.create_cuda_buffer:
                if self.weight_name is not None:
                    self.weight_cuda_buffer = weight_dict[self.weight_name].cuda().t()
                if self.bias_name is not None:
                    self.bias_cuda_buffer = weight_dict[self.bias_name].cuda()
            else:
                if self.weight_name is not None:
                    device = weight_dict[self.weight_name].device
                    if device.type in ["cuda", "mlu", "npu"]:
                        self.weight = weight_dict[self.weight_name]
                        if self.bias_name is not None:
                            self.bias = weight_dict[self.bias_name]
                        else:
                            self.bias = None
                    elif device.type == "cpu":
                        weight_shape = weight_dict[self.weight_name].shape
                        weight_dtype = weight_dict[self.weight_name].dtype
                        self.pin_weight = torch.empty(weight_shape, pin_memory=True, dtype=weight_dtype)
                        self.pin_weight.copy_(weight_dict[self.weight_name])

                        if self.bias_name is not None:
                            bias_shape = weight_dict[self.bias_name].shape
                            bias_dtype = weight_dict[self.bias_name].dtype
                            self.pin_bias = torch.empty(bias_shape, pin_memory=True, dtype=bias_dtype)
                            self.pin_bias.copy_(weight_dict[self.bias_name])
                        else:
                            self.bias = None
                            self.pin_bias = None
                        del weight_dict[self.weight_name]
                    else:
                        raise ValueError(f"Unsupported device type: {device.type}, only 'cpu' and 'cuda' are supported")
                else:
                    self.weight = None
                    self.bias = None

    def _calculate_size(self):
        if self.weight is None:
            return 0
        if self.bias is not None:
            return self.weight.numel() * self.weight.element_size() + self.bias.numel() * self.bias.element_size()
        return self.weight.numel() * self.weight.element_size()

    def clear(self):
        attrs = ["weight", "bias", "pinned_weight", "pinned_bias"]
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

    def to_cuda(self, non_blocking=False):
        if hasattr(self, "pin_weight") and self.pin_weight is not None:
            self.weight = self.pin_weight.cuda(non_blocking=non_blocking)
        else:
            self.weight = None
        if hasattr(self, "pin_bias") and self.pin_bias is not None:
            self.bias = self.pin_bias.cuda(non_blocking=non_blocking)
        else:
            self.bias = None

    def to_cpu(self, non_blocking=False):
        if hasattr(self, "pin_weight") and self.pin_weight is not None:
            self.weight = self.pin_weight.copy_(self.weight, non_blocking=non_blocking).cpu()
            if self.bias is not None:
                self.bias = self.pin_bias.copy_(self.bias, non_blocking=non_blocking).cpu()
        elif hasattr(self, "weight") and self.weight is not None:
            self.weight = self.weight.to("cpu", non_blocking=non_blocking)
            if hasattr(self, "bias") and self.bias is not None:
                self.bias = self.bias.to("cpu", non_blocking=non_blocking)

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        if self.weight_name is not None:
            destination[self.weight_name] = self.pin_weight if hasattr(self, "pin_weight") else self.weight
        if self.bias_name is not None:
            destination[self.bias_name] = self.pin_bias if hasattr(self, "pin_bias") else self.bias
        return destination

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        if self.weight_name is not None:
            if self.is_post_adapter:
                assert adapter_block_index is not None
                weight_name = re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", self.weight_name, count=1)
            else:
                weight_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.weight_name, count=1)

            if weight_name not in destination:
                self.weight = None
                return
            self.weight = self.weight_cuda_buffer.copy_(destination[weight_name], non_blocking=True)
        else:
            self.weight = None

        if self.bias_name is not None:
            bias_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.bias_name, count=1)
            self.bias = self.bias_cuda_buffer.copy_(destination[bias_name], non_blocking=True)
        else:
            self.bias = None


@LN_WEIGHT_REGISTER("Default")
class LNWeight(LNWeightTemplate):
    def __init__(self, weight_name=None, bias_name=None, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False, eps=1e-6):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter, eps)

    def load_from_disk(self):
        if self.weight_name is not None:
            if not torch._dynamo.is_compiling():
                self.weight = self.lazy_load_file.get_tensor(self.weight_name).to(GET_DTYPE()).pin_memory()
            else:
                self.weight = self.lazy_load_file.get_tensor(self.weight_name).to(GET_DTYPE())
        else:
            self.weight = None

        if self.bias_name is not None:
            if not torch._dynamo.is_compiling():
                self.bias = self.lazy_load_file.get_tensor(self.bias_name).to(GET_DTYPE()).pin_memory()
            else:
                self.bias = self.lazy_load_file.get_tensor(self.bias_name).to(GET_DTYPE())
        else:
            self.bias = None

    def apply(self, input_tensor):
        if self.sensitive_layer_dtype != self.infer_dtype:
            input_tensor = torch.nn.functional.layer_norm(
                input_tensor.float(),
                (input_tensor.shape[-1],),
                self.weight,
                self.bias,
                self.eps,
            ).to(self.infer_dtype)
        else:
            input_tensor = torch.nn.functional.layer_norm(input_tensor, (input_tensor.shape[-1],), self.weight, self.bias, self.eps)

        return input_tensor


@LN_WEIGHT_REGISTER("Triton")
class LNWeight(LNWeightTemplate):
    def __init__(self, weight_name=None, bias_name=None, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False, eps=1e-6):
        super().__init__(weight_name, bias_name, create_cuda_buffer, lazy_load, lazy_load_file, is_post_adapter, eps)

    def load_from_disk(self):
        if self.weight_name is not None:
            if not torch._dynamo.is_compiling():
                self.weight = self.lazy_load_file.get_tensor(self.weight_name).to(GET_DTYPE()).pin_memory()
            else:
                self.weight = self.lazy_load_file.get_tensor(self.weight_name).to(GET_DTYPE())
        else:
            self.weight = None

        if self.bias_name is not None:
            if not torch._dynamo.is_compiling():
                self.bias = self.lazy_load_file.get_tensor(self.bias_name).to(GET_DTYPE()).pin_memory()
            else:
                self.bias = self.lazy_load_file.get_tensor(self.bias_name).to(GET_DTYPE())
        else:
            self.bias = None

    def apply(self, input_tensor):
        input_tensor = norm_infer(input_tensor, self.weight, self.bias, self.eps)
        return input_tensor
