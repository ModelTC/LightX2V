import safetensors
import torch
from safetensors.torch import save_file

from lightx2v.utils.quant_utils import FloatQuantizer

model_path = "/data/nvme0/models/SekoTalk-Distill/audio_adapter_model.safetensors"

state_dict = {}
with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key)


new_state_dict = {}
new_model_path = "/data/nvme0/models/SekoTalk-Distill-fp8/audio_adapter_model_fp8.safetensors"

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

save_file(new_state_dict, new_model_path)
