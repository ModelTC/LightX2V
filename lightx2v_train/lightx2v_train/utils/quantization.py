from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from lightx2v_train.utils.registry import Register

QUANTIZER_REGISTER = Register()


def normalize_quant_config(config):
    if config is None:
        return None
    if not isinstance(config, Mapping) or set(config) - {"weight", "act"}:
        raise ValueError("quant must be a mapping containing weight and optional act.")
    if config.get("weight") is None:
        raise ValueError("quant.weight is required when QAT is enabled.")
    result = {}
    for role in ("weight", "act"):
        spec = config.get(role)
        if spec is None:
            result[role] = None
            continue
        if not isinstance(spec, Mapping) or set(spec) - {"bit", "symmetric", "granularity", "group_size"}:
            raise ValueError(f"Invalid quant.{role} fields.")
        bit, granularity = spec.get("bit"), spec.get("granularity")
        if not isinstance(bit, str) or bit not in QUANTIZER_REGISTER:
            raise ValueError(f"quant.{role}.bit must be int4 or nvfp4, got {bit!r}.")
        symmetric = spec.get("symmetric", True)
        if type(symmetric) is not bool:
            raise ValueError(f"quant.{role}.symmetric must be a boolean.")
        if bit == "nvfp4" and not symmetric:
            raise ValueError("NVFP4 only supports symmetric=true.")
        allowed = {"per_tensor", "per_group", "per_channel" if role == "weight" else "per_token"}
        if not isinstance(granularity, str) or granularity not in allowed:
            raise ValueError(f"Invalid quant.{role}.granularity={granularity!r}; expected {sorted(allowed)}.")
        normalized = {"bit": bit, "symmetric": symmetric, "granularity": granularity}
        if granularity == "per_group":
            group_size = spec.get("group_size")
            if type(group_size) is not int or group_size <= 0:
                raise ValueError(f"quant.{role}.group_size must be a positive integer for per_group.")
            normalized["group_size"] = group_size
        elif "group_size" in spec:
            raise ValueError(f"quant.{role}.group_size is only valid for per_group.")
        if bit == "nvfp4" and (granularity != "per_group" or normalized.get("group_size") != 16):
            raise ValueError("NVFP4 requires granularity=per_group and group_size=16.")
        result[role] = normalized
    return result


def validate_training_quantization(config):
    quant = normalize_quant_config(config.get("quant"))
    if quant is not None:
        model, training = config.get("model", {}), config.get("training", {})
        allowed = {"wan_t2v", "wan_t2v_14b", "minimax_h3_t2av"}
        if "cache_build" in config or training.get("method") != "dmd" or model.get("name") not in allowed or model.get("causal", False):
            raise ValueError("QAT supports only ordinary Wan/MiniMax-H3 DMD; phased and autoregressive DMD are not supported.")
        if training.get("student", {}).get("train_type") != "full":
            raise ValueError("QAT requires training.student.train_type=full; LoRA is not supported.")
    return quant


class _FakeQuantizer:
    def __init__(self, config):
        self.symmetric = config.get("symmetric", True)
        self.granularity = config["granularity"]
        self.group_size = config.get("group_size")

    def _view(self, x):
        if self.granularity == "per_tensor":
            return x.reshape(-1)
        if self.granularity == "per_group":
            if x.shape[-1] % self.group_size:
                raise ValueError(f"Last dimension {x.shape[-1]} is not divisible by group_size={self.group_size}.")
            return x.reshape(*x.shape[:-1], x.shape[-1] // self.group_size, self.group_size)
        return x

    def __call__(self, x):
        with torch.no_grad():
            value = self.quantize(x.detach().float()).to(x.dtype)
        return value + (x - x.detach())


@QUANTIZER_REGISTER("int4")
class INT4FakeQuantizer(_FakeQuantizer):
    def quantize(self, x):
        view = self._view(x)
        if self.symmetric:
            scale = view.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5) / 7.0
            return ((view / scale).round().clamp(-8, 7) * scale).reshape_as(x)
        minimum = view.amin(dim=-1, keepdim=True).clamp(max=0)
        maximum = view.amax(dim=-1, keepdim=True).clamp(min=0)
        scale = (maximum - minimum).clamp_min(1e-5) / 15.0
        zero_point = (-minimum / scale).round().clamp(0, 15)
        quantized = ((view / scale).round() + zero_point).clamp(0, 15)
        return ((quantized - zero_point) * scale).reshape_as(x)


@QUANTIZER_REGISTER("nvfp4")
class NVFP4FakeQuantizer(_FakeQuantizer):
    def quantize(self, x):
        view = self._view(x)
        absmax = x.abs().amax().clamp_min(1e-12)
        global_scale = 2688.0 / absmax
        scale = view.abs().amax(dim=-1, keepdim=True) / absmax * 448.0
        scale = scale.clamp(torch.finfo(torch.float8_e4m3fn).tiny, 448.0).to(torch.float8_e4m3fn).float()
        normalized = (view * global_scale / scale).clamp(-6.0, 6.0)
        magnitude = normalized.abs()
        step = torch.where(magnitude < 2, 0.5, torch.where(magnitude < 4, 1.0, 2.0))
        discrete = (magnitude / step).round() * step * normalized.sign()
        return (discrete * scale / global_scale).reshape_as(x)


class QATLinear(nn.Linear):
    def __init__(self, linear, quant_config):
        super().__init__(linear.in_features, linear.out_features, bias=linear.bias is not None, device="meta", dtype=linear.weight.dtype)
        self.weight = linear.weight
        self.bias = linear.bias
        self.quant_config = quant_config
        self.weight_quantizer = QUANTIZER_REGISTER[quant_config["weight"]["bit"]](quant_config["weight"])
        act = quant_config["act"]
        self.act_quantizer = None if act is None else QUANTIZER_REGISTER[act["bit"]](act)
        self.train(linear.training)

    def forward(self, input):
        if self.act_quantizer is not None:
            input = self.act_quantizer(input)
        return F.linear(input, self.weight_quantizer(self.weight), self.bias)

    def extra_repr(self):
        return f"{super().extra_repr()}, quant={self.quant_config}"


def replace_blocks_linear(model, quant_config):
    """Replace Linear layers in generative model blocks."""
    config = normalize_quant_config(quant_config)
    if config is None:
        return 0
    block_paths = ("blocks",) if hasattr(model, "blocks") else ("transformer_blocks", "token_refiner.refiner_blocks")
    replacements = []
    for name, module in model.named_modules():
        if not any(name.startswith(path + ".") for path in block_paths):
            continue
        if isinstance(module, QATLinear):
            if module.quant_config != config:
                raise ValueError(f"{name} already has a different QAT configuration.")
        elif isinstance(module, nn.Linear):
            parent_name, _, child_name = name.rpartition(".")
            replacements.append((model.get_submodule(parent_name), child_name, module))
    for parent, name, linear in replacements:
        setattr(parent, name, QATLinear(linear, config))
    return len(replacements)
