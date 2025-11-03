import argparse
import sys
from pathlib import Path

import safetensors
import torch
from safetensors.torch import save_file

sys.path.append(str(Path(__file__).parent.parent.parent))

from lightx2v.utils.quant_utils import FloatQuantizer
from tools.convert.quant import *


def main():
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(description="Quantize audio adapter model to FP8")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(project_root / "models" / "SekoTalk-Distill" / "audio_adapter_model.safetensors"),
        help="Path to input model file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(project_root / "models" / "SekoTalk-Distill-fp8" / "audio_adapter_model_fp8.safetensors"),
        help="Path to output quantized model file",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_path = Path(args.output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {}
    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    new_state_dict = {}

    for key in state_dict.keys():
        if key.startswith("ca") and ".to" in key and "weight" in key:
            print(f"Converting {key} to FP8, dtype: {state_dict[key].dtype}")

            ## fp8
            weight = state_dict[key].to(torch.float32).cuda()
            w_quantizer = FloatQuantizer("e4m3", True, "per_channel")
            weight, weight_scale, _ = w_quantizer.real_quant_tensor(weight)
            weight = weight.to(torch.float8_e4m3fn)
            weight_scale = weight_scale.to(torch.float32)

            ## QuantWeightMxFP4, QuantWeightMxFP6, QuantWeightMxFP8 for mxfp4,mxfp6,mxfp8
            # weight = state_dict[key].to(torch.bfloat16).cuda()
            # quantizer = QuantWeightMxFP4(weight)
            # weight, weight_scale, _ = quantizer.weight_quant_func(weight)

            new_state_dict[key] = weight.cpu()
            new_state_dict[key + "_scale"] = weight_scale.cpu()
        else:
            # 不匹配的权重转换为BF16
            print(f"Converting {key} to BF16, dtype: {state_dict[key].dtype}")
            new_state_dict[key] = state_dict[key].to(torch.bfloat16)

    save_file(new_state_dict, str(output_path))
    print(f"Quantized model saved to: {output_path}")


if __name__ == "__main__":
    main()
