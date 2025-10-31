from abc import ABCMeta

import torch
import torch.nn.functional as F

from lightx2v.utils.registry_factory import EMBEDDING_WEIGHT_REGISTER


class EmbeddingWeightTemplate(metaclass=ABCMeta):
    def __init__(self, weight_name, lazy_load=False, lazy_load_file=None):
        self.weight_name = weight_name
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

            del weight_dict[self.weight_name]

    def to_cpu(self, non_blocking=False):
        if hasattr(self, "pinned_weight"):
            self.weight = self.pinned_weight.copy_(self.weight, non_blocking=non_blocking).cpu()
        else:
            self.weight = self.weight.to("cpu", non_blocking=non_blocking)

    def to_cuda(self, non_blocking=False):
        self.weight = self.weight.cuda(non_blocking=non_blocking)


@EMBEDDING_WEIGHT_REGISTER("Default")
class EmbeddingWeight(EmbeddingWeightTemplate):
    def __init__(self, weight_name=None, lazy_load=False, lazy_load_file=None):
        super().__init__(weight_name, lazy_load, lazy_load_file)

    def apply(self, input_indices):
        output = F.embedding(input=input_indices, weight=self.weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)

        return output
