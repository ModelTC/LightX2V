"""Shape/dtype bindings for existing registered LightX2V operators.

Preserve checkpoint precision independently of the global DTYPE setting.
"""

import torch
from torch import nn

from lightx2v.utils.registry_factory import (
    ATTN_WEIGHT_REGISTER,
    CONV2D_WEIGHT_REGISTER,
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
)


class Linear(nn.Module):
    def __init__(self, weight, bias=None, native_requires_grad=False):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self._ops = {}
        self._unbiased_ops = {}
        self.native_requires_grad = native_requires_grad

    def forward(self, x):
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else self.weight.dtype
        if dtype not in self._ops:
            op = MM_WEIGHT_REGISTER["Default"]("weight", "bias" if self.bias is not None else None)
            tensors = {"weight": self.weight.to(dtype)}
            if self.bias is not None:
                tensors["bias"] = self.bias.to(dtype)
            op.load(tensors)
            self._ops[dtype] = op
        shape = x.shape[:-1]
        x = x.to(dtype)
        op = self._ops[dtype]
        # F.linear fuses bias for matrices and contiguous higher-rank inputs.
        # For strided higher-rank inputs it dispatches matmul then adds bias.
        # Flattening/copying those inputs changes both GEMM layout and BF16
        # rounding, so preserve that missing ranked-input path here.
        if x.ndim == 1 or (x.ndim != 2 and not x.is_contiguous()):
            # PyTorch's matmul folds a ranked input when its matrix operand
            # requires gradients, even inside no_grad/inference_mode. Preserve
            # this dispatch metadata without enabling gradients in LightX2V.
            # An autocast weight conversion under no_grad clears that flag.
            fold = self.native_requires_grad and dtype == self.weight.dtype and x.ndim > 2
            if fold:
                if dtype not in self._unbiased_ops:
                    plain = MM_WEIGHT_REGISTER["Default"]("weight")
                    plain.load({"weight": self.weight.to(dtype)})
                    self._unbiased_ops[dtype] = plain
                y = self._unbiased_ops[dtype].apply(x.reshape(-1, x.shape[-1])).reshape(*shape, self.weight.shape[0])
            else:
                y = torch.matmul(x, op.weight)
            if op.bias is not None:
                y = y + op.bias
            return y
        y = op.apply(x.reshape(-1, x.shape[-1]))
        return y.reshape(*shape, self.weight.shape[0])


class Conv2d(nn.Module):
    def __init__(self, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.op = CONV2D_WEIGHT_REGISTER["Default"]("weight", "bias" if bias is not None else None, stride=stride, padding=padding, dilation=dilation, groups=groups)
        self.op.load({"weight": weight, "bias": bias})
        self.register_buffer("weight", self.op.weight)
        self.register_buffer("bias", self.op.bias)

    def forward(self, x):
        return self.op.apply(x)


class LayerNorm(nn.Module):
    def __init__(self, weight=None, bias=None, eps=1e-5):
        super().__init__()
        self.op = LN_WEIGHT_REGISTER["torch"]("weight" if weight is not None else None, "bias" if bias is not None else None, eps=eps)
        self.op.load({"weight": weight, "bias": bias})
        self.register_buffer("weight", self.op.weight)
        self.register_buffer("bias", self.op.bias)

    def forward(self, x):
        self.op.infer_dtype = x.dtype
        self.op.sensitive_layer_dtype = x.dtype
        return self.op.apply(x)


class RMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.op = RMS_WEIGHT_REGISTER["torch"]("weight", eps=eps)
        self.op.load({"weight": weight})
        self.register_buffer("weight", self.op.weight)

    def forward(self, x):
        self.op.infer_dtype = x.dtype
        # RMSWeight's equal-dtype branch performs FP32 accumulation.
        self.op.sensitive_layer_dtype = x.dtype
        return self.op.apply(x)


class Attention:
    """Accept BHSD, preserve B=1, pass masks instead of silently dropping them."""

    def __init__(self):
        self.op = ATTN_WEIGHT_REGISTER["torch_sdpa"]()

    def __call__(self, q, k, v, mask=None, causal=False, scale=None):
        # Existing SDPA uses the standard scale. Preserve explicit CLIP scales.
        if scale is not None:
            q = q * (scale * q.shape[-1] ** 0.5)
        b, h, n, d = q.shape
        y = self.op.apply(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask, causal=causal)
        return y.reshape(b, n, h, d).transpose(1, 2)


def bind_modules(module):
    """Bind after strict state loading, while retaining checkpoint dtypes."""

    def visit(parent):
        for name, child in list(parent.named_children()):
            replacement = None
            if isinstance(child, nn.Linear):
                replacement = Linear(child.weight.detach(), None if child.bias is None else child.bias.detach(), child.weight.requires_grad)
            elif isinstance(child, nn.Conv2d):
                if child.padding_mode != "zeros":
                    raise ValueError("Only zero padded convolutions are supported")
                replacement = Conv2d(child.weight.detach(), None if child.bias is None else child.bias.detach(), child.stride, child.padding, child.dilation, child.groups)
            elif isinstance(child, nn.LayerNorm):
                replacement = LayerNorm(None if child.weight is None else child.weight.detach(), None if child.bias is None else child.bias.detach(), child.eps)
            elif type(child).__name__ == "RMSNorm" and hasattr(child, "weight"):
                replacement = RMSNorm(child.weight.detach(), child.eps)
            if replacement is None:
                visit(child)
            else:
                setattr(parent, name, replacement)

    visit(module)
