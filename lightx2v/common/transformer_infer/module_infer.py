"""Bind stateless inference methods to framework weight containers."""

from importlib import import_module

from lightx2v.common.modules.inference_weights import InferenceWeights
from lightx2v.common.modules.weight_module import WeightModuleList


def bind_inference(weights):
    if isinstance(weights, WeightModuleList):
        return tuple(bind_inference(child) for child in weights)
    if not isinstance(weights, InferenceWeights):
        return weights
    if weights.sequence:
        return SequentialInfer(weights)
    if weights.inference_type is None:
        return ModuleInfer(weights)
    module, name = weights.inference_type
    return getattr(import_module(module), name)(weights)


class ModuleInfer:
    def __init__(self, weights):
        self.weights = weights
        self._children = {name: bind_inference(child) for name, child in weights._modules.items()}

    def __getattr__(self, name):
        if name in self._children:
            return self._children[name]
        return getattr(self.weights, name)

    def __call__(self, *args, **kwargs):
        return self.infer(*args, **kwargs)


class SequentialInfer(ModuleInfer):
    def infer(self, x):
        for child in self._children.values():
            x = child(x)
        return x
