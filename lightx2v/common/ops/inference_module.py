"""Adapters for inference graphs whose checkpoint keys follow nn.Module names."""

import torch
from torch import nn

from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, CONV2D_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


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
        kwargs = {}
        if batch > 1:
            # FlashAttention's registry contract requires sequence boundaries
            # for multiple samples, including equal-length dense batches.
            kv_tokens = k.shape[1]
            kwargs = {
                "cu_seqlens_q": torch.arange(batch + 1, device=q.device, dtype=torch.int32) * tokens,
                "cu_seqlens_kv": torch.arange(batch + 1, device=k.device, dtype=torch.int32) * kv_tokens,
                "max_seqlen_q": tokens,
                "max_seqlen_kv": kv_tokens,
            }
        out = self.op.apply(q, k, v, **kwargs)
        return out.reshape(batch, tokens, heads, dim).transpose(1, 2)


def bind_linear_operators(module, backend="Default"):
    """Load registered dense linear operators after strict checkpoint loading."""
    for name, child in list(module.named_modules()):
        if type(child) is nn.Linear:
            module.set_submodule(name, InferenceLinear(child, name, backend))


def cast_matrix_weights(module, dtype=torch.bfloat16, extra_types=()):
    matrix_types = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Embedding) + extra_types
    for child in module.modules():
        if isinstance(child, matrix_types):
            child.to(dtype=dtype)


class InferenceLayerNorm(nn.Module):
    """Use the existing LN registry with the source module's dtype boundary."""

    def __init__(self, layer, name, backend="torch", fp32=False):
        super().__init__()
        self.fp32 = fp32
        weight_name = f"{name}.weight" if layer.weight is not None else None
        bias_name = f"{name}.bias" if layer.bias is not None else None
        self.op = LN_WEIGHT_REGISTER[backend](weight_name, bias_name, eps=layer.eps)
        state = {}
        if weight_name is not None:
            state[weight_name] = layer.weight.detach()
        if bias_name is not None:
            state[bias_name] = layer.bias.detach()
        self.op.load(state)

    def forward(self, x):
        dtype = x.dtype
        values = x.float() if self.fp32 else x
        # Preserve the original graph's per-layer policy, independent of the
        # global sensitive-layer setting used by other model families.
        self.op.infer_dtype = values.dtype
        self.op.sensitive_layer_dtype = values.dtype
        return self.op.apply(values).to(dtype)


class InferenceConv2d(nn.Module):
    def __init__(self, layer, name, backend="Default"):
        super().__init__()
        weight_name = f"{name}.weight"
        bias_name = f"{name}.bias" if layer.bias is not None else None
        self.op = CONV2D_WEIGHT_REGISTER[backend](weight_name, bias_name, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups)
        state = {weight_name: layer.weight.detach()}
        if bias_name is not None:
            state[bias_name] = layer.bias.detach()
        self.op.load(state)

    def forward(self, x):
        return self.op.apply(x)


def bind_spatial_operators(module, norm_backend="torch", conv_backend="Default", fp32_norm_types=()):
    for name, child in list(module.named_modules()):
        if type(child) is nn.LayerNorm or type(child) in fp32_norm_types:
            module.set_submodule(name, InferenceLayerNorm(child, name, norm_backend, fp32=type(child) in fp32_norm_types))
        elif type(child) is nn.Conv2d and child.padding_mode == "zeros":
            module.set_submodule(name, InferenceConv2d(child, name, conv_backend))
