import os
import re
from pathlib import Path

import torch
from safetensors import safe_open

from lightx2v.utils.envs import *
from lightx2v.utils.registry_factory import TENSOR_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


@TENSOR_REGISTER("Default")
class DefaultTensor:
    def __init__(self, tensor_name, create_cuda_buffer=False, create_cpu_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        self.tensor_name = tensor_name
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.is_post_adapter = is_post_adapter
        self.create_cuda_buffer = create_cuda_buffer
        self.create_cpu_buffer = create_cpu_buffer
        self.infer_dtype = GET_DTYPE()
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()

    def load(self, weight_dict):
        if self.create_cuda_buffer:
            self._load_cuda_buffer(weight_dict)
        elif self.create_cpu_buffer:
            self._load_cpu_pin_buffer()
        else:
            self._load_default_tensors(weight_dict)

    def _load_default_tensors(self, weight_dict):
        if not self.lazy_load:
            device = weight_dict[self.tensor_name].device
            if device.type == "cpu":
                tensor = weight_dict[self.tensor_name]
                self.pin_tensor = self._create_cpu_pin_tensor(tensor)
                del weight_dict[self.tensor_name]
            else:
                self.tensor = weight_dict[self.tensor_name]

    def _get_tensor(self, weight_dict=None, use_infer_dtype=False):
        if self.lazy_load:
            if Path(self.lazy_load_file).is_file():
                lazy_load_file_path = self.lazy_load_file
            else:
                lazy_load_file_path = os.path.join(self.lazy_load_file, f"block_{self.tensor_name.split('.')[1]}.safetensors")
            with safe_open(lazy_load_file_path, framework="pt", device="cpu") as lazy_load_file:
                tensor = lazy_load_file.get_tensor(self.tensor_name)
                if use_infer_dtype:
                    tensor = tensor.to(self.infer_dtype)
        else:
            tensor = weight_dict[self.tensor_name]
        return tensor

    def _create_cpu_pin_tensor(self, tensor):
        pin_tensor = torch.empty(tensor.shape, pin_memory=True, dtype=tensor.dtype)
        pin_tensor.copy_(tensor)
        del tensor
        return pin_tensor

    def _load_cuda_buffer(self, weight_dict):
        tensor = self._get_tensor(weight_dict, use_infer_dtype=self.lazy_load)
        self.tensor_cuda_buffer = tensor.to(AI_DEVICE)

    def _load_cpu_pin_buffer(self):
        tensor = self._get_tensor(use_infer_dtype=True)
        self.pin_tensor = self._create_cpu_pin_tensor(tensor)

    def to_cuda(self, non_blocking=False):
        self.tensor = self.pin_tensor.to(AI_DEVICE, non_blocking=non_blocking)

    def to_cpu(self, non_blocking=False):
        if hasattr(self, "pin_tensor"):
            self.tensor = self.pin_tensor.copy_(self.tensor, non_blocking=non_blocking).cpu()
        else:
            self.tensor = self.tensor.to("cpu", non_blocking=non_blocking)

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        destination[self.tensor_name] = self.pin_tensor if hasattr(self, "pin_tensor") else self.tensor
        return destination

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        if self.is_post_adapter:
            assert adapter_block_index is not None
            tensor_name = re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", self.tensor_name, count=1)
        else:
            tensor_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.tensor_name, count=1)
        if tensor_name not in destination:
            self.tensor = None
            return
        self.tensor = self.tensor_cuda_buffer.copy_(destination[tensor_name], non_blocking=True)

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        if self.is_post_adapter:
            assert adapter_block_index is not None
            self.tensor_name = re.sub(r"\.\d+", lambda m: f".{adapter_block_index}", self.tensor_name, count=1)
        else:
            self.tensor_name = re.sub(r"\.\d+", lambda m: f".{block_index}", self.tensor_name, count=1)
        if Path(self.lazy_load_file).is_file():
            lazy_load_file_path = self.lazy_load_file
        else:
            lazy_load_file_path = os.path.join(self.lazy_load_file, f"block_{block_index}.safetensors")
        with safe_open(lazy_load_file_path, framework="pt", device="cpu") as lazy_load_file:
            tensor = lazy_load_file.get_tensor(self.tensor_name).to(self.infer_dtype)
            self.pin_tensor = self.pin_tensor.copy_(tensor)
        del tensor
