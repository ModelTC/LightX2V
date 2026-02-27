import argparse

import safetensors.torch as st
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", type=str, required=True)
    parser.add_argument("--state-dict", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scale-const", type=float, default=2688.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    calib = torch.load(args.calib, map_location="cpu")
    state_dict = st.load_file(args.state_dict, device=str(device))

    new_dict = {}
    for key in state_dict.keys():
        if "weight_global_scale" in key:
            input_absmax = calib["absmax"][key.replace("weight_global_scale", "weight").replace("model.", "")]
            input_global_scale = (args.scale_const / input_absmax).to(torch.float32).to(device)
            weight_global_scale = state_dict[key].to(device)
            alpha = 1.0 / (input_global_scale * weight_global_scale)
            new_dict[key.replace("weight_global_scale", "alpha")] = alpha
            new_dict[key.replace("weight_global_scale", "input_global_scale")] = input_global_scale

    for key in new_dict.keys():
        state_dict[key] = new_dict[key]

    new_dict = {}
    for key in state_dict.keys():
        if "weight_global_scale" not in key:
            new_dict[key] = state_dict[key]

    st.save_file(tensors=new_dict, filename=args.output)


if __name__ == "__main__":
    main()
