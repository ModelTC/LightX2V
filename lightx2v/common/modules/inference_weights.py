"""Weight containers for checkpoint-compatible module declarations."""

from torch import nn

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList


class InferenceWeights(WeightModule):
    """Keep checkpoint storage separate from model inference methods.

    The nn.Module declarations are used only for strict checkpoint loading.
    Runtime containers retain tensors, registered operator leaves and metadata.
    """

    @classmethod
    def from_module(cls, source):
        if isinstance(source, nn.ModuleList):
            return WeightModuleList(cls.from_module(child) for child in source)
        if isinstance(source, nn.Sequential):
            result = cls()
            result.sequence = True
        elif hasattr(source, "_inference_type") or type(source) is nn.Module:
            result = cls()
            result.sequence = False
            result.inference_type = getattr(source, "_inference_type", None)
        else:
            # Operator leaves and external encoders keep their existing API.
            return source
        internal = set(vars(nn.Module()))
        for name, value in vars(source).items():
            if name not in internal or name == "training":
                setattr(result, name, value)
        for name, value in source._parameters.items():
            result.register_parameter(name, value)
        for name, value in source._buffers.items():
            result.register_parameter(name, value)
        for name, child in source._modules.items():
            result.add_module(name, cls.from_module(child) if child is not None else None)
        return result

    def parameters(self):
        for value in self._parameters.values():
            if value is not None:
                yield value
        for child in self._modules.values():
            if child is not None and hasattr(child, "parameters"):
                yield from child.parameters()
