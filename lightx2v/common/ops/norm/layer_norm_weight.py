from abc import ABCMeta, abstractmethod

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.common.ops.utils import *
from lightx2v.utils.envs import *
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER

from .triton_ops import norm_infer


class LNWeightTemplate(metaclass=ABCMeta):
    def __init__(
        self,
        weight_name=None,
        bias_name=None,
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
        self.bias_name = bias_name
        self.eps = eps
        self.create_cuda_buffer = create_cuda_buffer
        self.create_cpu_buffer = create_cpu_buffer
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.is_post_adapter = is_post_adapter
        self.config = {}
        self.infer_dtype = GET_DTYPE()
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()
        self.lora_prefix = lora_prefix
        self.lora_path = lora_path
        self.has_lora_branch = False
        self.has_diff = False
        self._get_base_attrs_mapping()
        self._get_lora_attr_mapping()

    def _get_base_attrs_mapping(self):
        self.base_attrs = []
        if self.weight_name is not None:
            self.base_attrs.append((self.weight_name, "weight", False))
        else:
            self.weight = None
        if self.bias_name is not None:
            self.base_attrs.append((self.bias_name, "bias", False))
        else:
            self.bias = None

    def _get_lora_attr_mapping(self):
        if self.weight_name is not None:
            _, _, _, self.weight_diff_name, self.bias_diff_name = build_lora_and_diff_names(self.weight_name, self.lora_prefix)
            self.lora_attrs = {
                "weight_diff": "weight_diff_name",
                "bias_diff": "bias_diff_name",
            }
        else:
            self.weight_diff_name = None
            self.bias_diff_name = None
            self.lora_attrs = {}

    def _get_actual_weight(self):
        if self.weight is None:
            return None
        if not hasattr(self, "weight_diff"):
            return self.weight
        return self.weight + self.weight_diff

    def _get_actual_bias(self):
        if self.bias is None:
            return None
        if not hasattr(self, "bias_diff"):
            return self.bias
        return self.bias + self.bias_diff

    def load(self, weight_dict):
        if not self.create_cuda_buffer and not self.create_cpu_buffer and not self.lazy_load:
            device_tensors, pin_tensors = create_default_tensors(self.base_attrs, weight_dict)
            self.weight = device_tensors.get("weight")
            self.bias = device_tensors.get("bias")
            self.pin_weight = pin_tensors.get("weight")
            self.pin_bias = pin_tensors.get("bias")
        elif self.create_cuda_buffer:
            result = create_cuda_buffers(
                self.base_attrs,
                weight_dict,
                self.lazy_load,
                self.lazy_load_file,
                use_infer_dtype=True,
            )
            self.weight_cuda_buffer = result.get("weight")
            self.bias_cuda_buffer = result.get("bias")
        elif self.create_cpu_buffer:
            result = create_cpu_buffers(self.base_attrs, self.lazy_load_file, use_infer_dtype=True)
            self.pin_weight = result.get("weight")
            self.pin_bias = result.get("bias")
            self.weight = None
            self.bias = None

    def register_diff(self, weight_dict):
        if not self.lazy_load or self.create_cuda_buffer or self.create_cpu_buffer:
            if self.weight_diff_name is not None and self.weight_diff_name in weight_dict:
                self.weight_diff = weight_dict[self.weight_diff_name]
                self.has_diff = True
                logger.debug(f"Register Diff to {self.weight_name}")
            if self.bias_diff_name is not None and self.bias_diff_name in weight_dict:
                self.bias_diff = weight_dict[self.bias_diff_name]
                self.has_diff = True
                logger.debug(f"Register Diff to {self.bias_name}")

    def set_config(self, config=None):
        if config is not None:
            self.config = config

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
        self.bias_diff_name = resolve_block_name(self.bias_diff_name, block_index)
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
        if self.weight_name is not None:
            if self.has_lora_branch or self.has_diff:
                self.load_lora_state_dict_from_disk(block_index)
            self.weight_name = resolve_block_name(self.weight_name, block_index, adapter_block_index, self.is_post_adapter)
            lazy_load_file_path = get_lazy_load_file_path(self.lazy_load_file, self.weight_name)
            if self.bias_name is not None:
                self.bias_name = resolve_block_name(
                    self.bias_name,
                    block_index,
                    adapter_block_index,
                    self.is_post_adapter,
                )
            with safe_open(lazy_load_file_path, framework="pt", device="cpu") as lazy_load_file:
                weight_tensor = lazy_load_file.get_tensor(self.weight_name).to(self.infer_dtype)
                self.pin_weight = self.pin_weight.copy_(weight_tensor)
                if self.bias_name is not None:
                    bias_tensor = lazy_load_file.get_tensor(self.bias_name).to(self.infer_dtype)
                    self.pin_bias = self.pin_bias.copy_(bias_tensor)
                else:
                    self.pin_bias = None
            del weight_tensor
        else:
            self.weight = None
            self.bias = None

    def to_cuda(self, non_blocking=False):
        move_attr_to_cuda(self, self.base_attrs, self.lora_attrs, non_blocking)

    def to_cpu(self, non_blocking=False):
        move_attr_to_cpu(self, self.base_attrs, self.lora_attrs, non_blocking)

    @abstractmethod
    def apply(self, input_tensor):
        pass


@LN_WEIGHT_REGISTER("Default")
class LNWeight(LNWeightTemplate):
    def __init__(
        self,
        weight_name=None,
        bias_name=None,
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
            bias_name,
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
        if self.sensitive_layer_dtype != self.infer_dtype:
            output_tensor = torch.nn.functional.layer_norm(
                input_tensor.float(),
                (input_tensor.shape[-1],),
                (self._get_actual_weight()),
                (self._get_actual_bias()),
                self.eps,
            ).to(self.infer_dtype)
        else:
            output_tensor = torch.nn.functional.layer_norm(
                input_tensor,
                (input_tensor.shape[-1],),
                (self._get_actual_weight()),
                (self._get_actual_bias()),
                self.eps,
            )

        return output_tensor


@LN_WEIGHT_REGISTER("Triton")
class LNWeight(LNWeightTemplate):
    def __init__(
        self,
        weight_name=None,
        bias_name=None,
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
            bias_name,
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
        output_tensor = norm_infer(
            input_tensor,
            (self._get_actual_weight()),
            self._get_actual_bias(),
            self.eps,
        )
        return output_tensor
