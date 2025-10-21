import argparse
import sys
from pathlib import Path

import safetensors
import torch
from safetensors.torch import save_file

sys.path.append(str(Path(__file__).parent.parent.parent))

from lightx2v.utils.quant_utils import FloatQuantizer


def main():
    parser = argparse.ArgumentParser(description="Quantize audio adapter model to FP8")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/nvme0/gushiqiao/models/Lightx2v_models/SekoTalk-Distill/audio_adapter_model.safetensors",
        help="Path to input model file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/data/nvme0/gushiqiao/models/Lightx2v_models/seko-new/SekoTalk-Distill-fp8/audio_adapter_model_fp8.safetensors",
        help="Path to output quantized model file",
    )
    args = parser.parse_args()

    model_path = args.model_path
    output_path = args.output_path

    state_dict = {}
    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    new_state_dict = {}

    for key in state_dict.keys():
        if key.startswith("ca") and ".to" in key and "weight" in key:
            print(key, state_dict[key].dtype)

            weight = state_dict[key].to(torch.float32).cuda()
            w_quantizer = FloatQuantizer("e4m3", True, "per_channel")
            weight, weight_scale, _ = w_quantizer.real_quant_tensor(weight)
            weight = weight.to(torch.float8_e4m3fn)
            weight_scale = weight_scale.to(torch.float32)

            new_state_dict[key] = weight.cpu()
            new_state_dict[key + "_scale"] = weight_scale.cpu()

    for key in state_dict.keys():
        if key not in new_state_dict.keys():
            new_state_dict[key] = state_dict[key]

    save_file(new_state_dict, output_path)
    print(f"Quantized model saved to: {output_path}")


if __name__ == "__main__":
    main()
