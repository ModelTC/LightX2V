import re
from abc import ABCMeta
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from lightx2v.utils.envs import *
from lightx2v.utils.registry_factory import EMBEDDING_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


class EmbeddingWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name, create_cuda_buffer=False, create_cpu_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        self.weight_name = weight_name
        self.create_cuda_buffer = create_cuda_buffer
        self.create_cpu_buffer = create_cpu_buffer
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.is_post_adapter = is_post_adapter
        self.infer_dtype = GET_DTYPE()
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
            if Path(self.lazy_load_file).is_file():
                lazy_load_file_path = self.lazy_load_file
            else:
                lazy_load_file_path = os.path.join(self.lazy_load_file, f"block_{self.weight_name.split('.')[1]}.safetensors")
            with safe_open(lazy_load_file_path, framework="pt", device="cpu") as lazy_load_file:
                tensor = lazy_load_file.get_tensor(self.weight_name)
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
        if Path(self.lazy_load_file).is_file():
            lazy_load_file_path = self.lazy_load_file
        else:
            lazy_load_file_path = os.path.join(self.lazy_load_file, f"block_{block_index}.safetensors")
        with safe_open(lazy_load_file_path, framework="pt", device="cpu") as lazy_load_file:
            weight_tensor = lazy_load_file.get_tensor(self.weight_name).to(self.infer_dtype)
            self.pin_weight = self.pin_weight.copy_(weight_tensor)
        del weight_tensor


@EMBEDDING_WEIGHT_REGISTER("Default")
class EmbeddingWeight(EmbeddingWeightTemplate):
    def __init__(self, weight_name=None, create_cuda_buffer=False, create_cpu_buffer=False, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file)

    def apply(self, input_indices):
        output = F.embedding(input=input_indices, weight=self.weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
        return output
