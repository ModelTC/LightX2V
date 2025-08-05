import torch

from lightx2v.utils.envs import *
from lightx2v.utils.registry_factory import TENSOR_REGISTER


@TENSOR_REGISTER("Default")
class DefaultTensor:
    def __init__(self, tensor_name, lazy_load=False, lazy_load_file=None):
        self.tensor_name = tensor_name
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file

    def load_from_disk(self):
        if not torch._dynamo.is_compiling():
            self.tensor = self.lazy_load_file.get_tensor(self.tensor_name).to(torch.bfloat16).pin_memory()
        else:
            self.tensor = self.lazy_load_file.get_tensor(self.tensor_name).to(torch.bfloat16)

    def load(self, weight_dict):
        if not self.lazy_load:
            self.tensor = weight_dict[self.tensor_name]
            self.pinned_tensor = torch.empty(self.tensor.shape, pin_memory=True, dtype=self.tensor.dtype)

    def clear(self):
        attrs = ["tensor", "pinned_tensor"]
        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)
                setattr(self, attr, None)

    def _calculate_size(self):
        return self.tensor.numel() * self.tensor.element_size()

    def to_cpu(self, non_blocking=False):
        if hasattr(self, "pinned_tensor"):
            self.tensor = self.pinned_tensor.copy_(self.tensor, non_blocking=non_blocking).cpu()
        else:
            self.tensor = self.tensor.to("cpu", non_blocking=non_blocking)

    def to_cuda(self, non_blocking=False):
        self.tensor = self.tensor.cuda(non_blocking=non_blocking)

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        destination[self.tensor_name] = self.tensor.cpu().detach().clone()
        return destination
