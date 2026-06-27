#!/bin/bash

# Wan2.1 Self-Forcing T2V 8-GPU Inference Script for NVIDIA
# System management interface: nvidia-smi

lightx2v_path=${LIGHTX2V_PATH:-/data/nvme1/wushuo/LightX2V}
model_path=${MODEL_PATH:-/data/nvme1/wushuo/hf_models/models/Wan2.1-T2V-1.3B}
gpus=${GPUS:-8}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

source "${lightx2v_path}/scripts/platforms/nvidia/dist/logging.sh"
source "${lightx2v_path}/scripts/base/base.sh"

mkdir -p "${lightx2v_path}/save_results"

torchrun --nproc_per_node="${gpus}" -m lightx2v.infer \
    --model_cls wan2.1_sf \
    --task t2v \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/nvidia/dist/wan_t2v_sf_dist8.json" \
    --prompt 'A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides.' \
    --save_result_path "${lightx2v_path}/save_results/output_lightx2v_wan_t2v_sf_dist8.mp4"
