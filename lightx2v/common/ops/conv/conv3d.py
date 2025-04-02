import torch
from abc import ABCMeta, abstractmethod
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
        self.weight = weight_dict[self.weight_name].cuda()
        self.bias = weight_dict[self.bias_name].cuda() if self.bias_name is not None else None

    def apply(self, input_tensor):
        input_tensor = torch.nn.functional.conv3d(input_tensor, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return input_tensor

    def to_cpu(self):
        self.weight = self.weight.cpu()
        if self.bias is not None:
            self.bias = self.bias.cpu()

    def to_cuda(self):
        self.weight = self.weight.cuda()
        if self.bias is not None:
            self.bias = self.bias.cuda()


@CONV3D_WEIGHT_REGISTER("Defaultt-Force-BF16")
class Conv3dWeightForceBF16(Conv3dWeight):
    def __init__(self, weight_name, bias_name, stride=1, padding=0, dilation=1, groups=1):
        super().__init__(weight_name, bias_name, stride, padding, dilation, groups)

    def load(self, weight_dict):
        self.weight = weight_dict[self.weight_name].to(torch.bfloat16).cuda()
        self.bias = weight_dict[self.bias_name].to(torch.bfloat16).cuda() if self.bias_name is not None else None
