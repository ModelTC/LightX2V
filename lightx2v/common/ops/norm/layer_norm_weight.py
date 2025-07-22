import torch
from abc import ABCMeta, abstractmethod
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER
from lightx2v.utils.envs import *
from apex.normalization.fused_layer_norm import fused_layer_norm, fused_layer_norm_affine


class LNWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name=None, bias_name=None, lazy_load=False, lazy_load_file=None, eps=1e-6):
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.eps = eps
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.config = {}

    def load(self, weight_dict):
        if not self.lazy_load:
            if self.weight_name is not None:
                self.weight = weight_dict[self.weight_name]
                self.pinned_weight = torch.empty(self.weight.shape, pin_memory=True, dtype=self.weight.dtype)
            else:
                self.weight = None
            if self.bias_name is not None:
                self.bias = weight_dict[self.bias_name]
                self.pinned_bias = torch.empty(self.bias.shape, pin_memory=True, dtype=self.bias.dtype)
            else:
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

    def to_cpu(self, non_blocking=False):
        if hasattr(self, "pinned_weight"):
            self.weight = self.pinned_weight.copy_(self.weight, non_blocking=non_blocking).cpu()
            if self.bias is not None:
                self.bias = self.pinned_bias.copy_(self.bias, non_blocking=non_blocking).cpu()
        else:
            if self.weight is not None:
                self.weight = self.weight.to("cpu", non_blocking=non_blocking)
            if self.bias is not None:
                self.bias = self.bias.to("cpu", non_blocking=non_blocking)

    def to_cuda(self, non_blocking=False):
        if self.weight is not None:
            self.weight = self.weight.cuda(non_blocking=non_blocking)
        if self.bias is not None:
            self.bias = self.bias.cuda(non_blocking=non_blocking)

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        if self.weight is not None:
            destination[self.weight_name] = self.weight.cpu().detach().clone()
        if self.bias is not None:
            destination[self.bias_name] = self.bias.cpu().detach().clone()
        return destination


@LN_WEIGHT_REGISTER("Default")
class LNWeight(LNWeightTemplate):
    def __init__(self, weight_name=None, bias_name=None, lazy_load=False, lazy_load_file=None, eps=1e-6):
        super().__init__(weight_name, bias_name, lazy_load, lazy_load_file, eps)

    def load_from_disk(self):
        if self.weight_name is not None:
            if not torch._dynamo.is_compiling():
                self.weight = self.lazy_load_file.get_tensor(self.weight_name).to(torch.bfloat16).pin_memory()
            else:
                self.weight = self.lazy_load_file.get_tensor(self.weight_name).to(torch.bfloat16)
        else:
            self.weight = None

        if self.bias_name is not None:
            if not torch._dynamo.is_compiling():
                self.bias = self.lazy_load_file.get_tensor(self.bias_name).to(torch.bfloat16).pin_memory()
            else:
                self.bias = self.lazy_load_file.get_tensor(self.bias_name).to(torch.bfloat16)
        else:
            self.bias = None

    def apply(self, input_tensor):
        if GET_DTYPE() != "BF16":
            input_tensor = torch.nn.functional.layer_norm(
                input_tensor.float(),
                (input_tensor.shape[-1],),
                self.weight,
                self.bias,
                self.eps,
            ).to(torch.bfloat16)
        else:
            input_tensor = torch.nn.functional.layer_norm(input_tensor, (input_tensor.shape[-1],), self.weight, self.bias, self.eps)
        return input_tensor
