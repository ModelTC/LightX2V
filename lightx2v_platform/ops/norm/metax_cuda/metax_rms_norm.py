from functools import lru_cache
from importlib import import_module

import torch

from lightx2v_platform.ops.norm.norm_template import RMSWeightTemplate
from lightx2v_platform.registry_factory import PLATFORM_RMS_WEIGHT_REGISTER


@lru_cache(maxsize=1)
def _load_metax_rms_norm_op():
    try:
        import_module("mcoplib._C")
        return torch.ops._C.rms_norm
    except (AttributeError, ImportError, OSError, RuntimeError) as error:
        raise RuntimeError("metax_rms_norm requires mcoplib._C with the _C::rms_norm operator") from error


@torch.library.custom_op(
    "lightx2v::metax_rms_norm",
    mutates_args=(),
    device_types="cuda",
)
def metax_rms_norm(input_tensor: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    output = torch.empty_like(input_tensor)
    _load_metax_rms_norm_op()(output, input_tensor, weight, eps)
    return output


@metax_rms_norm.register_fake
def _metax_rms_norm_fake(input_tensor: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(input_tensor)


@lru_cache(maxsize=1)
def _register_metax_rms_norm_lowering():
    from torch._inductor.lowering import add, mean, mul, register_lowering, rsqrt, to_dtype

    @register_lowering(torch.ops.lightx2v.metax_rms_norm.default, type_promotion_kind=None)
    def _metax_rms_norm_lowering(input_tensor, weight, eps):
        dtype = input_tensor.get_dtype()
        input_fp32 = to_dtype(input_tensor, torch.float32)
        variance = mean(mul(input_fp32, input_fp32), axis=-1, keepdim=True)
        output = mul(input_fp32, rsqrt(add(variance, eps)))
        output = mul(output, to_dtype(weight, torch.float32))
        return to_dtype(output, dtype)


@PLATFORM_RMS_WEIGHT_REGISTER("metax_rms_norm")
class MetaxRmsNormWeight(RMSWeightTemplate):
    def __init__(
        self,
        weight_name,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        eps=1e-6,
        lora_prefix="",
        lora_path="",
    ):
        super().__init__(
            weight_name,
            create_cuda_buffer,
            create_cpu_buffer,
            lazy_load,
            lazy_load_file,
            is_post_adapter,
            eps,
            lora_prefix,
            lora_path,
        )
        _load_metax_rms_norm_op()
        _register_metax_rms_norm_lowering()

    def _torch_fallback(self, input_tensor, weight):
        if self.sensitive_layer_dtype != self.infer_dtype:
            output = input_tensor * torch.rsqrt(input_tensor.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        else:
            input_fp32 = input_tensor.float()
            output = input_fp32 * torch.rsqrt(input_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = output.type_as(input_tensor)
        return output if weight is None else output * weight

    def apply(self, input_tensor):
        weight = getattr(self, "weight", None)
        valid_weight = weight is not None and weight.ndim == 1 and input_tensor.ndim > 0 and weight.numel() == input_tensor.shape[-1]
        use_metax_kernel = (
            valid_weight
            and input_tensor.numel() > 0
            and input_tensor.is_cuda
            and weight.device == input_tensor.device
            and input_tensor.dtype == weight.dtype
            and input_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and (self.sensitive_layer_dtype == self.infer_dtype or input_tensor.dtype == torch.float32)
        )
        if not use_metax_kernel:
            return self._torch_fallback(input_tensor, weight)

        original_shape = input_tensor.shape
        hidden_size = original_shape[-1]
        flat_input = input_tensor.contiguous().view(-1, hidden_size)
        output = metax_rms_norm(flat_input, weight.contiguous(), self.eps)
        return output.view(original_shape)


__all__ = ["MetaxRmsNormWeight"]
