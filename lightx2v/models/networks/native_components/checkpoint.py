"""Named tensor storage and operator binding; contains no model forward graph."""

import json
from pathlib import Path

import torch

from lightx2v.utils.registry_factory import EMBEDDING_WEIGHT_REGISTER
from lightx2v.utils.utils import load_safetensors_from_path

from .ops import Conv2d, LayerNorm, Linear, RMSNorm


def load_tensors(directory, dtype=None, device="cuda", variant=None):
    directory = Path(directory)
    index = sorted(directory.glob("*.safetensors.index.json"))
    if index:
        files = sorted(set(json.loads(index[0].read_text())["weight_map"].values()))
    else:
        files = sorted(directory.glob(f"*.{variant}.safetensors")) if variant else []
        files = files or sorted(p for p in directory.glob("*.safetensors") if ".fp16." not in p.name)
        if not files:
            bins = sorted(directory.glob("pytorch_model*.bin"))
            if len(bins) != 1:
                raise FileNotFoundError(f"No unambiguous local checkpoint in {directory}")
            return {k: v.to(device=device, dtype=dtype) for k, v in torch.load(bins[0], map_location="cpu", weights_only=True).items()}
    result = {}
    for file in files:
        path = directory / file if isinstance(file, str) else file
        tensors = load_safetensors_from_path(str(path))
        for k, v in tensors.items():
            if k in result:
                raise ValueError(f"Duplicate checkpoint key: {k}")
            result[k] = v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
    return result


class TensorWeights:
    def __init__(self, tensors, native_requires_grad=False):
        self.tensors = tensors
        self.operators = {}
        self.native_requires_grad = native_requires_grad

    def tensor(self, key):
        return self.tensors[key]

    def optional(self, key):
        return self.tensor(key) if key in self.tensors else None

    def embedding(self, name):
        if name not in self.operators:
            key = name + ".weight"
            op = EMBEDDING_WEIGHT_REGISTER["Default"](key)
            op.load({key: self.tensor(key)})
            self.operators[name] = op
        return self.operators[name]

    def linear(self, name):
        if name not in self.operators:
            self.operators[name] = Linear(self.tensor(name + ".weight"), self.optional(name + ".bias"), self.native_requires_grad)
        return self.operators[name]

    def conv(self, name, stride=1, padding=0):
        if name not in self.operators:
            self.operators[name] = Conv2d(self.tensor(name + ".weight"), self.optional(name + ".bias"), stride, padding)
        return self.operators[name]

    def norm(self, name, eps=1e-5):
        if name not in self.operators:
            self.operators[name] = LayerNorm(self.optional(name + ".weight"), self.optional(name + ".bias"), eps)
        return self.operators[name]

    def rms(self, name, eps=1e-6):
        if name not in self.operators:
            self.operators[name] = RMSNorm(self.tensor(name + ".weight"), eps)
        return self.operators[name]

    def group_norm(self, name):
        # There is no general GroupNorm registry in this base revision.
        # Weights stay separate from the model-specific group_norm call.
        return self.tensor(name + ".weight"), self.optional(name + ".bias")
