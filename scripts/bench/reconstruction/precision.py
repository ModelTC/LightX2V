"""Audit accelerator matrix operands during the excluded warmup request."""

from collections import Counter

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class BF16Audit(TorchDispatchMode):
    def __init__(self, device_type):
        super().__init__()
        self.device_type = device_type
        self.counts = Counter()
        self.violations = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        operands = ()
        operator = name.rsplit(".", 1)[0]
        if operator in {"aten.mm", "aten.bmm"}:
            operands = args[:2]
        elif operator in {"aten.addmm", "aten.baddbmm"}:
            operands = args[1:3]
        elif "convolution" in name or operator in {"aten.conv1d", "aten.conv2d", "aten.conv3d", "aten.conv_transpose2d", "aten.conv_transpose3d"}:
            operands = args[:2]
        elif "scaled_dot_product" in name:
            operands = args[:3]
        tensors = [value for value in operands if isinstance(value, torch.Tensor) and value.device.type == self.device_type and value.is_floating_point()]
        if tensors:
            signature = f"{name}:" + ",".join(str(value.dtype) for value in tensors)
            self.counts[signature] += 1
            if any(value.dtype != torch.bfloat16 for value in tensors):
                self.violations.append({"op": name, "dtypes": [str(value.dtype) for value in tensors], "shapes": [list(value.shape) for value in tensors]})
        return func(*args, **(kwargs or {}))

    def report(self):
        return {"matrix_calls": dict(self.counts), "violations": self.violations, "passed": bool(self.counts) and not self.violations}
