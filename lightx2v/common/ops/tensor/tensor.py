import re

import torch

from lightx2v.utils.envs import *
from lightx2v.utils.registry_factory import TENSOR_REGISTER


@TENSOR_REGISTER("Default")
class DefaultTensor:
    def __init__(self, tensor_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        self.tensor_name = tensor_name
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.is_post_adapter = is_post_adapter
        self.create_cuda_buffer = create_cuda_buffer
        self.infer_dtype = GET_DTYPE()
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()

    def load_from_disk(self):
        if not torch._dynamo.is_compiling():
            self.tensor = self.lazy_load_file.get_tensor(self.tensor_name).to(self.infer_dtype).pin_memory()
        else:
            self.tensor = self.lazy_load_file.get_tensor(self.tensor_name).to(self.infer_dtype)

    def load(self, weight_dict):
        if not self.lazy_load:
            if self.create_cuda_buffer:
                self.tensor_cuda_buffer = weight_dict[self.tensor_name].cuda()
            else:
                device = weight_dict[self.tensor_name].device
                if device.type == "cuda":
                    self.tensor = weight_dict[self.tensor_name]
                elif device.type == "cpu":
                    tensor_shape = weight_dict[self.tensor_name].shape
                    tensor_dtype = weight_dict[self.tensor_name].dtype
                    self.pin_tensor = torch.empty(tensor_shape, pin_memory=True, dtype=tensor_dtype)
                    self.pin_tensor.copy_(weight_dict[self.tensor_name])
                    del weight_dict[self.tensor_name]
                else:
                    raise ValueError(f"Unsupported device type: {device.type}, only 'cpu' and 'cuda' are supported")

    def clear(self):
        attrs = ["tensor", "pinned_tensor"]
        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)
                setattr(self, attr, None)

    def _calculate_size(self):
        return self.tensor.numel() * self.tensor.element_size()

    def to_cuda(self, non_blocking=False):
        self.tensor = self.pin_tensor.cuda(non_blocking=non_blocking)

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
