"""Adapters for inference graphs whose checkpoint keys follow nn.Module names."""

import torch
from torch import nn

from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


def validate_bf16_graph_config(config):
    if GET_DTYPE() != torch.bfloat16:
        raise ValueError("This inference graph requires DTYPE=BF16")
    unsupported = [key for key in ("parallel", "cpu_offload", "lazy_load", "unload_modules", "dit_quantized", "lora_configs") if config.get(key)]
    if unsupported:
        raise NotImplementedError(f"Unsupported inference settings: {', '.join(unsupported)}")


class InferenceLinear(nn.Module):
    def __init__(self, layer, name, backend):
        super().__init__()
        self.out_features = layer.out_features
        bias_name = f"{name}.bias" if layer.bias is not None else None
        self.op = MM_WEIGHT_REGISTER[backend](f"{name}.weight", bias_name)
        state = {f"{name}.weight": layer.weight.detach()}
        if bias_name is not None:
            state[bias_name] = layer.bias.detach()
        self.op.load(state)
        if self.op.weight.dtype != torch.bfloat16:
            raise ValueError(f"Linear backend {backend!r} does not preserve BF16 weights for {name}")

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1]).to(self.op.weight.dtype).contiguous()
        return self.op.apply(x).reshape(*shape[:-1], self.out_features)


class InferenceAttention(nn.Module):
    """Bridge BHSD model tensors to the registered BSHD attention contract."""

    def __init__(self, backend):
        super().__init__()
        self.op = ATTN_WEIGHT_REGISTER[backend]()

    def forward(self, q, k, v):
        batch, heads, tokens, dim = q.shape
        q, k, v = (x.to(v.dtype).transpose(1, 2).contiguous() for x in (q, k, v))
        out = self.op.apply(q, k, v)
        return out.reshape(batch, tokens, heads, dim).transpose(1, 2)


class InferenceSparseLinear(InferenceLinear):
    def forward(self, x):
        return x.replace(super().forward(x.feats))


def bind_linear_operators(module, backend="Default", prefix=""):
    """Load registered linear operators after strict checkpoint loading."""
    from lightx2v.common.ops.sparse.linear import SparseLinear

    for name, child in list(module.named_children()):
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, SparseLinear):
            setattr(module, name, InferenceSparseLinear(child, path, backend))
        elif type(child) is nn.Linear:
            setattr(module, name, InferenceLinear(child, path, backend))
        else:
            bind_linear_operators(child, backend, path)


def cast_matrix_weights(module, dtype=torch.bfloat16, extra_types=()):
    matrix_types = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Embedding) + extra_types
    for child in module.modules():
        if isinstance(child, matrix_types):
            child.to(dtype=dtype)
