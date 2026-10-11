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


def _gcn_arch():
    try:
        return torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    except Exception:
        return ""


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
        if not hasattr(torch, "_scaled_mm"):
            raise RuntimeError("fp8-rocm requires torch._scaled_mm, which is absent from this PyTorch/ROCm build.")
        arch = _gcn_arch()
        if not arch.startswith("gfx1201"):
            raise RuntimeError(f"fp8-rocm (torch._scaled_mm) is validated only on gfx1201 (RDNA4); detected '{arch}'.")

    def load(self, weight_dict):
        weight = weight_dict[self.weight_name].to(AI_DEVICE).to(torch.float32)  # (N, K)
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / self.FP8_MAX  # (N, 1)
        self.weight = (weight / scale).clamp(-self.FP8_MAX, self.FP8_MAX).to(torch.float8_e4m3fn)
        self.weight_scale = scale.to(torch.float32)
        del weight
        # Drop the source weight so it does not coexist with every quantized one.
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
        if not hasattr(torch, "_int_mm"):
            raise RuntimeError("int8-rocm requires torch._int_mm, which is absent from this PyTorch/ROCm build.")

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
            x_q = (input_tensor / x_scale).round().clamp(-self.INT8_MAX, self.INT8_MAX).to(torch.int8).contiguous()  # (M, K)
            out = torch._int_mm(x_q, self.weight.t().contiguous())  # (M, N) int32
            out = (out.to(torch.float32) * x_scale * self.weight_scale.t()).to(input_tensor.dtype)  # dequant
        if self.bias is not None:
            out = out + self.bias
        return out
