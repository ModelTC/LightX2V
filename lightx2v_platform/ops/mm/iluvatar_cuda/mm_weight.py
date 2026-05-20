import torch

from lightx2v.utils.quant_utils import IntegerQuantizer
from lightx2v_platform.ops.mm.template import MMWeightQuantTemplate
from lightx2v_platform.registry_factory import PLATFORM_MM_WEIGHT_REGISTER

try:
    import ixformer.inference.functions as ixf
except ImportError:
    ixf = None


@PLATFORM_MM_WEIGHT_REGISTER("int8-iluvatar")
class MMWeightWint8channelAint8channeldynamicIluvatar(MMWeightQuantTemplate):
    """
    Name: W-int8-channel-sym-A-int8-channel-sym-dynamic-iluvatar

    Quant MM:
        Weight: int8 perchannel sym
        Act: int8 perchannel dynamic sym
        Kernel: iluvatar
    """

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
        assert ixf is not None, "iluvatar ixformer is not installed."
        self.load_func = self.load_int8_perchannel_sym
        self.weight_need_transpose = False
        self.act_quant_func = self.act_quant_int8_perchannel_sym_iluvatar

    def _ensure_int8_weight_and_scale(self, weight_dict):
        """Fill missing weight_scale (or int8 weight) so load_quantized can run.

        Some quantized checkpoints omit per-layer scales (e.g. adaLN) or use alternate
        key names; others keep a few layers in float — per-channel int8 + scale is
        then derived to match ixformer w8a8.
        """
        if self.lazy_load:
            return
        if self.weight_name not in weight_dict:
            return
        if self.weight_scale_name in weight_dict:
            return
        base = self.weight_name.removesuffix(".weight")
        for alt in (f"{base}.scale", f"{self.weight_name}_scale"):
            if alt in weight_dict:
                weight_dict[self.weight_scale_name] = weight_dict[alt].float()
                return
        w = weight_dict[self.weight_name]
        if w.dtype in (torch.float16, torch.bfloat16, torch.float32):
            w_float = w.to(torch.float32)
            w_quantizer = IntegerQuantizer(8, True, "per_channel")
            qw, scale, _ = w_quantizer.real_quant_tensor(w_float)
            dev = w.device
            weight_dict[self.weight_name] = qw.to(torch.int8).to(dev)
            weight_dict[self.weight_scale_name] = scale.to(torch.float32).to(dev)

    def load(self, weight_dict):
        self._ensure_int8_weight_and_scale(weight_dict)
        super().load(weight_dict)

    def act_quant_int8_perchannel_sym_iluvatar(self, x):
        device = x.device
        input_tensor_quant = torch.empty(x.shape, dtype=torch.int8, device=device)
        input_tensor_scale = torch.empty(x.shape[:-1], dtype=torch.float32, device=device)
        ixf.dynamic_scaled_int8_quant(output=input_tensor_quant, input=x, scale=input_tensor_scale)
        return input_tensor_quant, input_tensor_scale

    def apply(self, input_tensor):
        squeeze_output = False
        dtype = input_tensor.dtype
        if input_tensor.dim() == 3 and input_tensor.shape[0] == 1:
            input_tensor = input_tensor.squeeze(0)
            squeeze_output = True
        input_tensor_quant, input_tensor_scale = self.act_quant_int8_perchannel_sym_iluvatar(input_tensor)
        output = ixf.w8a8(input=input_tensor_quant, weight=self.weight, i_scales=input_tensor_scale, w_scales=self.weight_scale.reshape(-1), bias=self.bias, out_dtype=dtype)
        if squeeze_output:
            output = output.unsqueeze(0)
        return output
