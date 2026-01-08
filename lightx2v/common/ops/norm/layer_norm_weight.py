import os
import re
from abc import ABCMeta, abstractmethod
from pathlib import Path

import torch
from safetensors import safe_open
from loguru import logger

from lightx2v.utils.envs import *
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

from .triton_ops import norm_infer


class LNWeightTemplate(metaclass=ABCMeta):
    def __init__(
        self,
        weight_name=None,
        bias_name=None,
        weight_diff_name=None,
        bias_diff_name=None,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        eps=1e-6,
    ):
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.weight_diff_name = weight_diff_name
        self.bias_diff_name = bias_diff_name
        self.weight_diff = 0.0
        self.bias_diff = 0.0
        self.eps = eps
        self.create_cuda_buffer = create_cuda_buffer
        self.create_cpu_buffer = create_cpu_buffer
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.is_post_adapter = is_post_adapter
        self.config = {}
        self.infer_dtype = GET_DTYPE()
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()

    def register_diff(self, weight_dict):
        if self.weight_diff_name is None and self.bias_diff_name is None:
            return
        self.weight_diff = weight_dict[self.weight_diff_name]
        self.bias_diff = weight_dict[self.bias_diff_name]
        logger.debug(f"Register Diff to {self.weight_name}")

    def load(self, weight_dict):
        if self.create_cuda_buffer:
            self._load_cuda_buffers(weight_dict)
        elif self.create_cpu_buffer:
            self._load_cpu_pin_buffers()
        else:
            self._load_default_tensors(weight_dict)

    def _load_default_tensors(self, weight_dict):
        if not self.lazy_load and self.weight_name is not None:
            device = weight_dict[self.weight_name].device
            if device.type == "cpu":
                weight_tensor = weight_dict[self.weight_name]
                self.pin_weight = self._create_cpu_pin_tensor(weight_tensor)
                bias_tensor = (
                    weight_dict[self.bias_name] if self.bias_name is not None else None
                )
                self.pin_bias = (
                    self._create_cpu_pin_tensor(bias_tensor)
                    if bias_tensor is not None
                    else None
                )
                self.bias = None
                del weight_dict[self.weight_name]
            else:
                self.weight = weight_dict[self.weight_name]
                self.bias = (
                    weight_dict[self.bias_name] if self.bias_name is not None else None
                )
        else:
            self.weight = None
            self.bias = None

    def _get_tensor(self, name, weight_dict=None, use_infer_dtype=False):
        if name is None:
            return None
        if self.lazy_load:
            if Path(self.lazy_load_file).is_file():
                lazy_load_file_path = self.lazy_load_file
            else:
                lazy_load_file_path = os.path.join(
                    self.lazy_load_file, f"block_{name.split('.')[1]}.safetensors"
                )
            with safe_open(
                lazy_load_file_path, framework="pt", device="cpu"
            ) as lazy_load_file:
                tensor = lazy_load_file.get_tensor(name)
                if use_infer_dtype:
                    tensor = tensor.to(self.infer_dtype)
        else:
            tensor = weight_dict[name]
        return tensor

    def _create_cpu_pin_tensor(self, tensor):
        if tensor is None:
            return None
        pin_tensor = torch.empty(tensor.shape, pin_memory=True, dtype=tensor.dtype)
        pin_tensor.copy_(tensor)
        del tensor
        return pin_tensor

    def _load_cuda_buffers(self, weight_dict):
        weight_tensor = self._get_tensor(
            self.weight_name, weight_dict, use_infer_dtype=self.lazy_load
        )
        if weight_tensor is not None:
            self.weight_cuda_buffer = weight_tensor.to(AI_DEVICE)

        bias_tensor = self._get_tensor(
            self.bias_name, weight_dict, use_infer_dtype=self.lazy_load
        )
        if bias_tensor is not None:
            self.bias_cuda_buffer = bias_tensor.to(AI_DEVICE)

    def _load_cpu_pin_buffers(self):
        weight_tensor = self._get_tensor(self.weight_name, use_infer_dtype=True)
        if weight_tensor is not None:
            self.pin_weight = self._create_cpu_pin_tensor(weight_tensor)
        else:
            self.weight = None

        bias_tensor = self._get_tensor(self.bias_name, use_infer_dtype=True)
        if bias_tensor is not None:
            self.pin_bias = self._create_cpu_pin_tensor(bias_tensor)
        else:
            self.bias = None
            self.pin_bias = None

    @abstractmethod
    def apply(self, input_tensor):
        pass

    def set_config(self, config=None):
        if config is not None:
            self.config = config

    def to_cuda(self, non_blocking=False):
        if hasattr(self, "pin_weight") and self.pin_weight is not None:
            self.weight = self.pin_weight.to(AI_DEVICE, non_blocking=non_blocking)
        else:
            self.weight = None
        if hasattr(self, "pin_bias") and self.pin_bias is not None:
            self.bias = self.pin_bias.to(AI_DEVICE, non_blocking=non_blocking)
        else:
            self.bias = None

    def to_cpu(self, non_blocking=False):
        if hasattr(self, "pin_weight") and self.pin_weight is not None:
            self.weight = self.pin_weight.copy_(
                self.weight, non_blocking=non_blocking
            ).cpu()
            if self.bias is not None:
                self.bias = self.pin_bias.copy_(
                    self.bias, non_blocking=non_blocking
                ).cpu()
        elif hasattr(self, "weight") and self.weight is not None:
            self.weight = self.weight.to("cpu", non_blocking=non_blocking)
            if hasattr(self, "bias") and self.bias is not None:
                self.bias = self.bias.to("cpu", non_blocking=non_blocking)

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        if self.weight_name is not None:
            destination[self.weight_name] = (
                self.pin_weight if hasattr(self, "pin_weight") else self.weight
            )
        if self.bias_name is not None:
            destination[self.bias_name] = (
                self.pin_bias if hasattr(self, "pin_bias") else self.bias
            )
        return destination

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        if self.weight_name is not None:
            if self.is_post_adapter:
                assert adapter_block_index is not None
                weight_name = re.sub(
                    r"\.\d+",
                    lambda m: f".{adapter_block_index}",
                    self.weight_name,
                    count=1,
                )
            else:
                weight_name = re.sub(
                    r"\.\d+", lambda m: f".{block_index}", self.weight_name, count=1
                )

            if weight_name not in destination:
                self.weight = None
                return
            self.weight = self.weight_cuda_buffer.copy_(
                destination[weight_name], non_blocking=True
            )
        else:
            self.weight = None

        if self.bias_name is not None:
            bias_name = re.sub(
                r"\.\d+", lambda m: f".{block_index}", self.bias_name, count=1
            )
            self.bias = self.bias_cuda_buffer.copy_(
                destination[bias_name], non_blocking=True
            )
        else:
            self.bias = None

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        if self.weight_name is not None:
            if Path(self.lazy_load_file).is_file():
                lazy_load_file_path = self.lazy_load_file
            else:
                lazy_load_file_path = os.path.join(
                    self.lazy_load_file, f"block_{block_index}.safetensors"
                )
            if self.is_post_adapter:
                self.weight_name = re.sub(
                    r"\.\d+",
                    lambda m: f".{adapter_block_index}",
                    self.weight_name,
                    count=1,
                )
            else:
                self.weight_name = re.sub(
                    r"\.\d+", lambda m: f".{block_index}", self.weight_name, count=1
                )

            with safe_open(
                lazy_load_file_path, framework="pt", device="cpu"
            ) as lazy_load_file:
                weight_tensor = lazy_load_file.get_tensor(self.weight_name).to(
                    self.infer_dtype
                )
                self.pin_weight = self.pin_weight.copy_(weight_tensor)
            del weight_tensor

        if self.bias_name is not None:
            if Path(self.lazy_load_file).is_file():
                lazy_load_file_path = self.lazy_load_file
            else:
                lazy_load_file_path = os.path.join(
                    self.lazy_load_file, f"block_{block_index}.safetensors"
                )
            if self.is_post_adapter:
                assert adapter_block_index is not None
                self.bias_name = re.sub(
                    r"\.\d+",
                    lambda m: f".{adapter_block_index}",
                    self.bias_name,
                    count=1,
                )
            else:
                self.bias_name = re.sub(
                    r"\.\d+", lambda m: f".{block_index}", self.bias_name, count=1
                )

            with safe_open(
                lazy_load_file_path, framework="pt", device="cpu"
            ) as lazy_load_file:
                bias_tensor = lazy_load_file.get_tensor(self.bias_name).to(
                    self.infer_dtype
                )
                self.pin_bias.copy_(bias_tensor)
            del bias_tensor


@LN_WEIGHT_REGISTER("Default")
class LNWeight(LNWeightTemplate):
    def __init__(
        self,
        weight_name=None,
        bias_name=None,
        weight_diff_name=None,
        bias_diff_name=None,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        eps=1e-6,
    ):
        super().__init__(
            weight_name,
            bias_name,
            weight_diff_name,
            bias_diff_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            eps,
        )

    def apply(self, input_tensor):
        if self.sensitive_layer_dtype != self.infer_dtype:
            output_tensor = torch.nn.functional.layer_norm(
                input_tensor.float(),
                (input_tensor.shape[-1],),
                (
                    self.weight
                    if self.weight_diff_name is None
                    else self.weight + self.weight_diff
                ),
                (
                    self.bias
                    if self.bias_diff_name is None
                    else self.bias + self.bias_diff
                ),
                self.eps,
            ).to(self.infer_dtype)
        else:
            output_tensor = torch.nn.functional.layer_norm(
                input_tensor,
                (input_tensor.shape[-1],),
                (
                    self.weight
                    if self.weight_diff_name is None
                    else self.weight + self.weight_diff
                ),
                (
                    self.bias
                    if self.bias_diff_name is None
                    else self.bias + self.bias_diff
                ),
                self.eps,
            )

        return output_tensor


@LN_WEIGHT_REGISTER("Triton")
class LNWeight(LNWeightTemplate):
    def __init__(
        self,
        weight_name=None,
        bias_name=None,
        weight_diff_name=None,
        bias_diff_name=None,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        eps=1e-6,
    ):
        super().__init__(
            weight_name,
            bias_name,
            weight_diff_name,
            bias_diff_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            eps,
        )

    def apply(self, input_tensor):
        output_tensor = norm_infer(
            input_tensor,
            (
                self.weight
                if self.weight_diff_name is None
                else self.weight + self.weight_diff
            ),
            self.bias if self.bias_diff_name is None else self.bias + self.bias_diff,
            self.eps,
        )
        return output_tensor
