###  Using this script to convert ViGen-DiT Lora Format to Lightx2v
###
###  Cmd line:python convert_vigen_to_x2v_lora.py model_lora.pt model_lora_converted.safetensors
###
###  ViGen-DiT Project Url: https://github.com/yl-1993/ViGen-DiT
###
import os
import sys

import torch
from safetensors.torch import load_file, save_file

if len(sys.argv) != 3:
    print("用法: python convert_lora.py <输入文件> <输出文件.safetensors>")
    sys.exit(1)

ckpt_path = sys.argv[1]
output_path = sys.argv[2]

if not os.path.exists(ckpt_path):
    print(f"❌ 输入文件不存在: {ckpt_path}")
    sys.exit(1)

if ckpt_path.endswith(".safetensors"):
    state_dict = load_file(ckpt_path)
else:
    state_dict = torch.load(ckpt_path, map_location="cpu")

if "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
elif "model" in state_dict:
    state_dict = state_dict["model"]

mapped_dict = {}

# 映射表定义
attn_map = {
    "attn1": "self_attn",
    "attn2": "cross_attn",
}
proj_map = {
    "to_q": "q",
    "to_k": "k",
    "to_v": "v",
    "to_out": "o",
    "add_k_proj": "k_img",
    "add_v_proj": "v_img",
}
lora_map = {
    "lora_A": "lora_down",
    "lora_B": "lora_up",
}

for k, v in state_dict.items():
    # 预处理：将 to_out.0 / to_out.1 统一替换为 to_out
    k = k.replace("to_out.0", "to_out").replace("to_out.1", "to_out")
    k = k.replace(".default", "")  # 去除.default

    parts = k.split(".")

    # === Attention Blocks ===
    if k.startswith("blocks.") and len(parts) >= 5:
        block_id = parts[1]

        if parts[2].startswith("attn"):
            attn_raw = parts[2]
            proj_raw = parts[3]
            lora_raw = parts[4]

            if attn_raw in attn_map and proj_raw in proj_map and lora_raw in lora_map:
                attn_name = attn_map[attn_raw]
                proj_name = proj_map[proj_raw]
                lora_name = lora_map[lora_raw]
                new_k = f"diffusion_model.blocks.{block_id}.{attn_name}.{proj_name}.{lora_name}.weight"
                mapped_dict[new_k] = v
                continue
            else:
                print(f"无法映射 attention key: {k}")
                continue
        # === FFN Blocks ===
        elif parts[2] == "ffn":
            if parts[3:6] == ["net", "0", "proj"]:
                layer_id = "0"
                lora_raw = parts[6]
            elif parts[3:5] == ["net", "2"]:
                layer_id = "2"
                lora_raw = parts[5]
            else:
                print(f"无法解析 FFN key: {k}")
                continue

            if lora_raw not in lora_map:
                print(f"未知 FFN LoRA 类型: {k}")
                continue

            lora_name = lora_map[lora_raw]
            new_k = f"diffusion_model.blocks.{block_id}.ffn.{layer_id}.{lora_name}.weight"
            mapped_dict[new_k] = v
            continue
    # === Text Embedding ===
    elif k.startswith("condition_embedder.text_embedder.linear_"):
        layer_id = parts[2].split("_")[1]
        lora_raw = parts[3]
        if lora_raw in lora_map:
            lora_name = lora_map[lora_raw]
            new_k = f"diffusion_model.text_embedding.{layer_id}.{lora_name}.weight"
            mapped_dict[new_k] = v
            continue
        else:
            print(f"text_embedder 未知 LoRA 类型: {k}")
            continue
    """
    # === Time Embedding ===
    elif k.startswith("condition_embedder.time_embedder.linear_"):
        layer_id = parts[2].split("_")[1]
        lora_raw = parts[3]
        if lora_raw in lora_map:
            lora_name = lora_map[lora_raw]
            new_k = f"diffusion_model.time_embedding.{layer_id}.{lora_name}.weight"
            mapped_dict[new_k] = v
            continue
        else:
            print(f"time_embedder 未知 LoRA 类型: {k}")
            continue

    # === Time Projection ===
    elif k.startswith("condition_embedder.time_proj."):
        lora_raw = parts[2]
        if lora_raw in lora_map:
            lora_name = lora_map[lora_raw]
            new_k = f"diffusion_model.time_projection.1.{lora_name}.weight"
            mapped_dict[new_k] = v
            continue
        else:
            print(f"time_proj 未知 LoRA 类型: {k}")
            continue
    """
    # fallback
    print(f"未识别结构 key: {k}")

# 保存
print(f"\n✅ 成功重命名 {len(mapped_dict)} 个 LoRA 参数")
save_file(mapped_dict, output_path)
print(f"💾 已保存为: {output_path}")
