from abc import ABCMeta, abstractmethod

import torch
from loguru import logger

from lightx2v.common.ops.utils import *
from lightx2v.utils.envs import *
from lightx2v.utils.registry_factory import CONV3D_WEIGHT_REGISTER


class Conv3dWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name, bias_name, stride=1, padding=0, dilation=1, groups=1, lora_prefix="diffusion_model.blocks"):
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.config = {}
        self.lora_prefix = lora_prefix
        self.has_lora_branch = False
        self.has_diff = False
        self._get_base_attrs_mapping()
        self._get_lora_attr_mapping()

    def _get_base_attrs_mapping(self):
        self.base_attrs = []
        self.base_attrs.append((self.weight_name, "weight", False))
        self.base_attrs.append((self.bias_name, "bias", False))

    def _get_lora_attr_mapping(self):
        _, _, _, self.weight_diff_name, self.bias_diff_name = build_lora_and_diff_names(self.weight_name, self.lora_prefix)
        self.lora_attrs = {
            "weight_diff": "weight_diff_name",
            "bias_diff": "bias_diff_name",
        }

    def _get_actual_weight(self):
        if not hasattr(self, "weight_diff"):
            return self.weight
        return self.weight + self.weight_diff

    def _get_actual_bias(self, bias=None):
        if bias is not None:
            if not hasattr(self, "bias_diff"):
                return bias
            return bias + self.bias_diff
        else:
            if not hasattr(self, "bias") or self.bias is None:
                return None
            if not hasattr(self, "bias_diff"):
                return self.bias
            return self.bias + self.bias_diff

    def register_diff(self, weight_dict):
        if self.weight_diff_name in weight_dict:
            self.weight_diff = weight_dict[self.weight_diff_name]
            logger.debug(f"Register Diff to {self.weight_name}")
        if self.bias_diff_name in weight_dict:
            self.bias_diff = weight_dict[self.bias_diff_name]
            logger.debug(f"Register Diff to {self.bias_name}")

    def set_config(self, config=None):
        if config is not None:
            self.config = config

    @abstractmethod
    def load(self, weight_dict):
        pass

    @abstractmethod
    def apply(self, input_tensor):
        pass


@CONV3D_WEIGHT_REGISTER("Default")
class Conv3dWeight(Conv3dWeightTemplate):
    def __init__(self, weight_name, bias_name, stride=1, padding=0, dilation=1, groups=1, lora_prefix="diffusion_model.blocks"):
        super().__init__(weight_name, bias_name, stride, padding, dilation, groups, lora_prefix)

    def load(self, weight_dict):
        device_tensors, pin_tensors = create_default_tensors(self.base_attrs, weight_dict)
        self.weight = device_tensors.get("weight")
        self.bias = device_tensors.get("bias")
        self.pin_weight = pin_tensors.get("weight")
        self.pin_bias = pin_tensors.get("bias")

    def apply(self, input_tensor):
        output_tensor = torch.nn.functional.conv3d(
            input_tensor,
            weight=self._get_actual_weight(),
            bias=self._get_actual_bias(),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return output_tensor

    def to_cuda(self, non_blocking=False):
        move_attr_to_cuda(self, self.base_attrs, self.lora_attrs, non_blocking)

    def to_cpu(self, non_blocking=False):
        move_attr_to_cpu(self, self.base_attrs, self.lora_attrs, non_blocking)

    def state_dict(self, destination=None):
        return state_dict(self, self.base_attrs, self.lora_attrs, destination)

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        return load_state_dict(self, self.base_attrs, self.lora_attrs, destination, block_index, adapter_block_index)
