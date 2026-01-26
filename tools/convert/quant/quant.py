from abc import ABCMeta

import torch

from lightx2v.utils.registry_factory import CONVERT_WEIGHT_REGISTER

try:
    from qtorch.quant import float_quantize
except ImportError:
    float_quantize = None

try:
    from lightx2v_kernel.gemm import scaled_mxfp4_quant, scaled_mxfp6_quant, scaled_mxfp8_quant, scaled_nvfp4_quant
except ImportError:
    pass


class QuantTemplate(metaclass=ABCMeta):
    def __init__(self, weight):
        if weight.dim() != 2:
            raise ValueError(f"Only 2D tensors supported. Got {weight.dim()}D tensor")
        if torch.isnan(weight).any():
            raise ValueError("Tensor contains NaN values")

        self.weight_quant_func = None
        self.extra = {}


@CONVERT_WEIGHT_REGISTER("int8")
class QuantWeightINT8(QuantTemplate):
    def __init__(self, weight):
        super().__init__(weight)
        self.weight_quant_func = self.load_int8_weight

    @torch.no_grad()
    def load_int8_weight(self, w, comfyui_mode=False):
        org_w_shape = w.shape
        if not comfyui_mode:
            max_val = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        else:
            max_val = w.abs().max()
        qmin, qmax = -128, 127
        scales = max_val / qmax
        w_q = torch.clamp(torch.round(w / scales), qmin, qmax).to(torch.int8)

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w_q).sum() == 0

        if not comfyui_mode:
            scales = scales.view(org_w_shape[0], -1)
            w_q = w_q.reshape(org_w_shape)

        return w_q, scales, self.extra


@CONVERT_WEIGHT_REGISTER("fp8")
class QuantWeightFP8(QuantTemplate):
    def __init__(self, weight):
        super().__init__(weight)
        if float_quantize is None:
            raise ImportError("qtorch is required for FP8 quantization. Please install it with: pip install qtorch")
        self.weight_quant_func = self.load_fp8_weight

    @torch.no_grad()
    def load_fp8_weight(self, w, comfyui_mode=False):
        org_w_shape = w.shape
        if not comfyui_mode:
            max_val = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        else:
            max_val = w.abs().max()
        finfo = torch.finfo(torch.float8_e4m3fn)
        qmin, qmax = finfo.min, finfo.max
        scales = max_val / qmax
        scaled_tensor = w / scales
        scaled_tensor = torch.clip(scaled_tensor, qmin, qmax)
        w_q = float_quantize(scaled_tensor.float(), 4, 3, rounding="nearest").to(torch.float8_e4m3fn)

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w_q).sum() == 0

        if not comfyui_mode:
            scales = scales.view(org_w_shape[0], -1)
            w_q = w_q.reshape(org_w_shape)

        return w_q, scales, self.extra


@CONVERT_WEIGHT_REGISTER("mxfp4")
class QuantWeightMxFP4(QuantTemplate):
    def __init__(self, weight):
        super().__init__(weight)
        self.weight_quant_func = self.load_mxfp4_weight

    @torch.no_grad()
    def load_mxfp4_weight(self, w, comfyui_mode=False):
        device = w.device
        w = w.cuda().to(torch.bfloat16)
        w_q, scales = scaled_mxfp4_quant(w)
        w_q, scales = w_q.to(device), scales.to(device)
        return w_q, scales, self.extra


@CONVERT_WEIGHT_REGISTER("mxfp6")
class QuantWeightMxFP6(QuantTemplate):
    def __init__(self, weight):
        super().__init__(weight)
        self.weight_quant_func = self.load_mxfp6_weight

    @torch.no_grad()
    def load_mxfp6_weight(self, w, comfyui_mode=False):
        device = w.device
        w = w.cuda().to(torch.bfloat16)
        w_q, scales = scaled_mxfp6_quant(w)
        w_q, scales = w_q.to(device), scales.to(device)
        return w_q, scales, self.extra


@CONVERT_WEIGHT_REGISTER("mxfp8")
class QuantWeightMxFP8(QuantTemplate):
    def __init__(self, weight):
        super().__init__(weight)
        self.weight_quant_func = self.load_mxfp8_weight

    @torch.no_grad()
    def load_mxfp8_weight(self, w, comfyui_mode=False):
        device = w.device
        w = w.cuda().to(torch.bfloat16)
        w_q, scales = scaled_mxfp8_quant(w)
        w_q, scales = w_q.to(device), scales.to(device)
        return w_q, scales, self.extra


@CONVERT_WEIGHT_REGISTER("nvfp4")
class QuantWeightNVFP4(QuantTemplate):
    def __init__(self, weight):
        super().__init__(weight)
        self.weight_quant_func = self.load_fp4_weight

    @torch.no_grad()
    def load_fp4_weight(self, w, comfyui_mode=False):
        device = w.device
        w = w.cuda().to(torch.bfloat16)
        weight_global_scale = (2688.0 / torch.max(torch.abs(w))).to(torch.float32)
        w_q, scales = scaled_nvfp4_quant(w, weight_global_scale)
        w_q, scales = w_q.to(device), scales.to(device)
        self.extra["weight_global_scale"] = weight_global_scale.to(device)
        return w_q, scales, self.extra
