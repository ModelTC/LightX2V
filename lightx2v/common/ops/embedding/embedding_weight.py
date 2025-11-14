import re
from abc import ABCMeta

import torch
import torch.nn.functional as F

from lightx2v.utils.registry_factory import EMBEDDING_WEIGHT_REGISTER


class EmbeddingWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name, create_cuda_buffer=False, lazy_load=False, lazy_load_file=None, is_post_adapter=False):
        self.weight_name = weight_name
        self.create_cuda_buffer = create_cuda_buffer
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file
        self.is_post_adapter = is_post_adapter
        self.config = {}

    def load(self, weight_dict):
        if not self.lazy_load:
            if self.create_cuda_buffer:
                self.weight_cuda_buffer = weight_dict[self.weight_name].cuda()
            else:
                device = weight_dict[self.weight_name].device
                if device.type == "cuda":
                    self.weight = weight_dict[self.weight_name]
                elif device.type == "cpu":
                    weight_shape = weight_dict[self.weight_name].shape
                    weight_dtype = weight_dict[self.weight_name].dtype
                    self.pin_weight = torch.empty(weight_shape, pin_memory=True, dtype=weight_dtype)
                    self.pin_weight.copy_(weight_dict[self.weight_name])
                    del weight_dict[self.weight_name]
                else:
                    raise ValueError(f"Unsupported device type: {device.type}, only 'cpu' and 'cuda' are supported")

    def to_cuda(self, non_blocking=False):
        self.weight = self.pin_weight.cuda(non_blocking=non_blocking)

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


@EMBEDDING_WEIGHT_REGISTER("Default")
class EmbeddingWeight(EmbeddingWeightTemplate):
    def __init__(self, weight_name=None, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, lazy_load, lazy_load_file)

    def apply(self, input_indices):
        output = F.embedding(input=input_indices, weight=self.weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)

        return output
