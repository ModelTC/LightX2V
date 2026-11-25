"""Convert exactly the DiT projections that the existing FP8 preset quantizes."""
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from lightx2v.common.ops.mm.mm_weight import scaled_nvfp4_quant

model = Path(sys.argv[1])
out = Path(sys.argv[2])
source = model / "dit-5b/diffusion_pytorch_model.safetensors"
reference = model / "quantized/dit-5b-fp8-f16-accum-qmax14.safetensors"

weights = {}
count = 0
with safe_open(source, framework="pt", device="cpu") as original, safe_open(reference, framework="pt", device="cpu") as fp8:
    for name in original.keys():
        tensor = original.get_tensor(name)
        scale_name = name + "_scale"
        if not name.endswith(".weight") or scale_name not in fp8.keys():
            weights[name] = tensor
            continue
        weight = tensor.to(device="cuda", dtype=torch.bfloat16)
        global_scale = 2688.0 / weight.abs().amax().float().clamp_min(1e-12)
        packed, scales = scaled_nvfp4_quant(weight, global_scale)
        weights[name] = packed.cpu()
        weights[scale_name] = scales.cpu()
        weights[name.removesuffix(".weight") + ".input_global_scale"] = torch.ones(())
        # The benchmark multiplies this reciprocal by the activation dequantization factor.
        weights[name.removesuffix(".weight") + ".alpha"] = global_scale.reciprocal().cpu()
        count += 1
assert count > 0
temporary = out.with_suffix(".tmp")
save_file(weights, temporary, metadata={"activation_scale": "dynamic", "source": str(source)})
temporary.replace(out)
print(f"Converted {count} DiT projections to {out}", flush=True)
