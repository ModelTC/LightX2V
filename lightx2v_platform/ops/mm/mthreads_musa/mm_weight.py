"""LightX2V MM weight wrapper for MUSA FP8 scaled-MM."""

import torch

from lightx2v_platform.ops.mm.template import MMWeightPerChannelQuantTemplate
from lightx2v_platform.registry_factory import PLATFORM_MM_WEIGHT_REGISTER

from .fp8_scaled_mm import fp8_scaled_mm, per_token_quant_fp8


@PLATFORM_MM_WEIGHT_REGISTER("fp8-musa")
class MMWeightWfp8channelAfp8tokendynamicMusa(MMWeightPerChannelQuantTemplate):
    """W8A8 FP8 linear with per-channel weights and per-token activations."""

    checkpoint_dtype = torch.float8_e4m3fn

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
        self.load_func = self.load_fp8_perchannel_sym
        self.weight_need_transpose = True
        self.act_quant_func = per_token_quant_fp8

    def apply(self, input_tensor):
        if input_tensor.ndim < 2:
            raise ValueError(f"input_tensor must have at least 2 dimensions, got {tuple(input_tensor.shape)}")

        input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
        input_quant, input_scale = self.act_quant_func(input_2d)
        output_dtype = input_tensor.dtype if input_tensor.dtype in (torch.bfloat16, torch.float16) else self.infer_dtype
        bias = self.bias if hasattr(self, "bias") else None
        output_2d = fp8_scaled_mm(
            input_quant,
            self.weight,
            input_scale,
            self.weight_scale.float(),
            out_dtype=output_dtype,
            bias=bias,
        )
        return output_2d.reshape(*input_tensor.shape[:-1], self.weight.shape[1])


__all__ = ["MMWeightWfp8channelAfp8tokendynamicMusa"]
