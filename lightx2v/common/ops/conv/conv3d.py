from abc import ABCMeta, abstractmethod

import torch

from lightx2v.utils.registry_factory import CONV3D_WEIGHT_REGISTER


class Conv3dWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name, bias_name, stride=1, padding=0, dilation=1, groups=1):
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.config = {}

    @abstractmethod
    def load(self, weight_dict):
        pass

    @abstractmethod
    def apply(self, input_tensor):
        pass

    def set_config(self, config=None):
        if config is not None:
            self.config = config


@CONV3D_WEIGHT_REGISTER("Default")
class Conv3dWeight(Conv3dWeightTemplate):
    def __init__(self, weight_name, bias_name, stride=1, padding=0, dilation=1, groups=1):
        super().__init__(weight_name, bias_name, stride, padding, dilation, groups)

    def load(self, weight_dict):
        device = weight_dict[self.weight_name].device
        if device.type == "cpu":
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
            self.weight = weight_dict[self.weight_name]
            if self.bias_name is not None:
                self.bias = weight_dict[self.bias_name]
            else:
                self.bias = None

    def apply(self, input_tensor):
        input_tensor = torch.nn.functional.conv3d(
            input_tensor,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return input_tensor

    def to_cuda(self, non_blocking=False):
        self.weight = self.pin_weight.cuda(non_blocking=non_blocking)
        if hasattr(self, "pin_bias") and self.pin_bias is not None:
            self.bias = self.pin_bias.cuda(non_blocking=non_blocking)

    def to_cpu(self, non_blocking=False):
        if hasattr(self, "pin_weight"):
            self.weight = self.pin_weight.copy_(self.weight, non_blocking=non_blocking).cpu()
            if self.bias is not None:
                self.bias = self.pin_bias.copy_(self.bias, non_blocking=non_blocking).cpu()
        else:
            self.weight = self.weight.to("cpu", non_blocking=non_blocking)
            if hasattr(self, "bias") and self.bias is not None:
                self.bias = self.bias.to("cpu", non_blocking=non_blocking)

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        destination[self.weight_name] = self.pin_weight if hasattr(self, "pin_weight") else self.weight  # .cpu().detach().clone().contiguous()
        if self.bias_name is not None:
            destination[self.bias_name] = self.pin_bias if hasattr(self, "pin_bias") else self.bias  # .cpu().detach().clone()
        return destination

    def clear(self):
        attrs = ["weight", "bias", "pinned_weight", "pinned_bias"]
        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)
                setattr(self, attr, None)
