"""AMD ROCm quantized GEMM weights (aiter-free).

FP8 (RDNA4 / gfx1201) drives torch._scaled_mm; INT8 (RDNA3 / gfx1100) drives
torch._int_mm. Both quantize the released BF16 checkpoint on load, so no
pre-quantized checkpoint and no aiter build are required. CUTLASS / sgl / vLLM
FP8 kernels are CUDA-only, which is why these platform-native paths exist.
"""

import torch

from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v_platform.ops.mm.template import MMWeightQuantTemplate
from lightx2v_platform.registry_factory import PLATFORM_MM_WEIGHT_REGISTER


@PLATFORM_MM_WEIGHT_REGISTER("fp8-rocm")
class MMWeightWfp8channelAfp8channeldynamicRocm(MMWeightQuantTemplate):
    """W-fp8-e4m3-channel-sym / A-fp8-e4m3-token-dynamic via torch._scaled_mm.

    MIOpen/CUTLASS/sgl FP8 kernels are unavailable on ROCm gfx1201 (RDNA4), but
    torch._scaled_mm drives a working FP8 GEMM there. Weights are quantized from
    the released BF16 checkpoint on load (no pre-quantized checkpoint needed).
    """

    FP8_MAX = 448.0

    def __init__(
        self,
        weight_name,
        bias_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            bias_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            lora_prefix,
            lora_path,
        )
        # Weight stored (N, K) fp8; a column-major (K, N) view is taken in apply().
        self.weight_need_transpose = False

    def load(self, weight_dict):
        weight = weight_dict[self.weight_name].to(AI_DEVICE).to(torch.float32)  # (N, K)
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / self.FP8_MAX  # (N, 1)
        self.weight = (weight / scale).clamp(-self.FP8_MAX, self.FP8_MAX).to(torch.float8_e4m3fn)
        self.weight_scale = scale.to(torch.float32)
        del weight
        # Release the BF16 source immediately so it does not coexist with every
        # quantized weight (halves the load-time peak on GPU).
        weight_dict.pop(self.weight_name, None)
        if self.bias_name is not None:
            self.bias = weight_dict[self.bias_name].to(AI_DEVICE).to(self.infer_dtype)
        else:
            self.bias = None

    def apply(self, input_tensor):
        x_scale = (input_tensor.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / self.FP8_MAX).float()  # (M, 1)
        x_q = (input_tensor / x_scale).clamp(-self.FP8_MAX, self.FP8_MAX).to(torch.float8_e4m3fn)  # (M, K)
        output_tensor = torch._scaled_mm(
            x_q,
            self.weight.t(),  # (K, N) column-major
            scale_a=x_scale,
            scale_b=self.weight_scale.t(),  # (1, N)
            bias=self.bias,
            out_dtype=input_tensor.dtype,
        )
        return output_tensor


class MMWeightFusedFp8Rocm(MMWeightWfp8channelAfp8channeldynamicRocm):
    """Fuse several row-concatenatable linears (q/k/v, gate/up) into one fp8 GEMM.

    Per-output-channel weight scales make this numerically identical to the
    separate GEMMs, while cutting GEMM count and quantizing the (shared) input
    activation once instead of once per linear. ``apply()`` returns the
    concatenated output; the caller splits it with ``split_sizes``.
    """

    def __init__(self, weight_names, bias_name=None, **kwargs):
        super().__init__(weight_names[0], bias_name, **kwargs)
        self.fused_weight_names = list(weight_names)
        self.split_sizes = None

    def load(self, weight_dict):
        weights = [weight_dict[name].to(AI_DEVICE).to(torch.float32) for name in self.fused_weight_names]  # each (N_i, K)
        self.split_sizes = [w.shape[0] for w in weights]
        weight = torch.cat(weights, dim=0)  # (sum_N, K)
        del weights
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / self.FP8_MAX  # (sum_N, 1)
        self.weight = (weight / scale).clamp(-self.FP8_MAX, self.FP8_MAX).to(torch.float8_e4m3fn)
        self.weight_scale = scale.to(torch.float32)
        del weight
        for name in self.fused_weight_names:
            weight_dict.pop(name, None)
        self.bias = None


@PLATFORM_MM_WEIGHT_REGISTER("int8-rocm")
class MMWeightWint8channelAint8tokenRocm(MMWeightQuantTemplate):
    """W-int8-channel-sym / A-int8-token-dynamic via torch._int_mm.

    RDNA3 (gfx1100, e.g. W7900) has no FP8 tensor path, but torch._int_mm drives
    a working INT8 WMMA GEMM there. Weights are quantized from the released BF16
    checkpoint on load (no prequantized checkpoint needed).
    """

    INT8_MAX = 127.0

    def __init__(
        self,
        weight_name,
        bias_name=None,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="diffusion_model.blocks",
        lora_path="",
    ):
        super().__init__(weight_name, bias_name, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file, is_post_adapter, lora_prefix, lora_path)
        self.weight_need_transpose = False

    def load(self, weight_dict):
        weight = weight_dict[self.weight_name].to(AI_DEVICE).to(torch.float32)  # (N, K)
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / self.INT8_MAX  # (N, 1)
        self.weight = (weight / scale).round().clamp(-self.INT8_MAX, self.INT8_MAX).to(torch.int8)
        self.weight_scale = scale.to(torch.float32)
        del weight
        weight_dict.pop(self.weight_name, None)
        if self.bias_name is not None:
            self.bias = weight_dict[self.bias_name].to(AI_DEVICE).to(self.infer_dtype)
        else:
            self.bias = None

    def apply(self, input_tensor):
        if input_tensor.shape[0] <= 16:
            # torch._int_mm requires M > 16; dequantize to BF16 for the small
            # prefill segments (rare, negligible cost).
            weight = (self.weight.to(torch.float32) * self.weight_scale).to(input_tensor.dtype)  # (N, K)
            out = torch.mm(input_tensor, weight.t())
        else:
            x_scale = (input_tensor.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / self.INT8_MAX).float()  # (M, 1)
            x_q = (input_tensor / x_scale).round().clamp(-self.INT8_MAX, self.INT8_MAX).to(torch.int8)  # (M, K)
            out = torch._int_mm(x_q, self.weight.t())  # (M, N) int32
            out = (out.to(torch.float32) * x_scale * self.weight_scale.t()).to(input_tensor.dtype)  # dequant
        if self.bias is not None:
            out = out + self.bias
        return out


class MMWeightFusedInt8Rocm(MMWeightWint8channelAint8tokenRocm):
    """Fuse q/k/v, gate/up into one INT8 GEMM; apply() returns the concatenated
    output, split by ``split_sizes``. See MMWeightFusedFp8Rocm."""

    def __init__(self, weight_names, bias_name=None, **kwargs):
        super().__init__(weight_names[0], bias_name, **kwargs)
        self.fused_weight_names = list(weight_names)
        self.split_sizes = None

    def load(self, weight_dict):
        weights = [weight_dict[name].to(AI_DEVICE).to(torch.float32) for name in self.fused_weight_names]
        self.split_sizes = [w.shape[0] for w in weights]
        weight = torch.cat(weights, dim=0)
        del weights
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / self.INT8_MAX
        self.weight = (weight / scale).round().clamp(-self.INT8_MAX, self.INT8_MAX).to(torch.int8)
        self.weight_scale = scale.to(torch.float32)
        del weight
        for name in self.fused_weight_names:
            weight_dict.pop(name, None)
        self.bias = None
