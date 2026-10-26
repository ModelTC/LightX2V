"""Convert original Wan UMT5 linears to the FP8-F16 reduced-range contract."""

import re

import torch

from lightx2v.utils.fp8_f16_accum_checkpoint import ACTIVATION_QMAX, WEIGHT_QMAX
from lightx2v.utils.fp8_f16_accum_checkpoint import T5_QUANTIZATION_PROFILE as QUANTIZATION_PROFILE

_LINEAR = re.compile(r"blocks\.(\d+)\.(attn\.[qkvo]|ffn\.(?:fc[12]|gate\.0))\.weight")
_BLOCK_LINEARS = {"attn.q", "attn.k", "attn.v", "attn.o", "ffn.fc1", "ffn.fc2", "ffn.gate.0"}


class T5FP8F16AccumQuantization:
    def __init__(self, model_type):
        if model_type != "wan_t5":
            raise ValueError(f"{QUANTIZATION_PROFILE} requires model_type=wan_t5")
        self.linears = {}

    @property
    def metadata(self):
        return {
            "format": "pt",
            "quantization_profile": QUANTIZATION_PROFILE,
            "weight_qmax": str(WEIGHT_QMAX),
            "activation_qmax": str(ACTIVATION_QMAX),
            "quantized_linears": str(sum(map(len, self.linears.values()))),
        }

    def quantize_weight(self, name, weight, default_quantize):
        match = _LINEAR.fullmatch(name)
        if match is None:
            raise ValueError(f"Unexpected UMT5 linear for {QUANTIZATION_PROFILE}: {name}")
        block, linear = match.groups()
        seen = self.linears.setdefault(int(block), set())
        if linear in seen:
            raise ValueError(f"UMT5 linear is duplicated: {name}")
        if weight.ndim != 2 or weight.numel() == 0 or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError(f"{name} must be an original nonempty floating-point matrix, got {weight.dtype} {tuple(weight.shape)}")
        values = weight.to(torch.float32, copy=True)
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} contains nonfinite values")
        scales = values.abs().amax(dim=1, keepdim=True).clamp_min_(1e-8).div_(WEIGHT_QMAX)
        quantized = values.div_(scales).clamp_(-WEIGHT_QMAX, WEIGHT_QMAX).to(torch.float8_e4m3fn)
        seen.add(linear)
        return quantized, scales, {}

    def validate(self):
        if not self.linears:
            raise ValueError("No UMT5 linears were selected for FP8-F16 accumulation conversion")
        for block, linears in self.linears.items():
            missing = _BLOCK_LINEARS - linears
            if missing:
                raise ValueError(f"UMT5 block {block} is missing linears: {sorted(missing)}")
