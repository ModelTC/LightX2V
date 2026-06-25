#!/bin/bash

# Qwen Image I2I 8-GPU Inference Script for NVIDIA
# System management interface: nvidia-smi

lightx2v_path=${LIGHTX2V_PATH:-/data/nvme1/wushuo/LightX2V}
model_path=${MODEL_PATH:-/data/nvme1/wushuo/hf_models/models/Qwen-Image-2512}
gpus=${GPUS:-8}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

source "${lightx2v_path}/scripts/platforms/nvidia/dist_8/logging.sh"
source "${lightx2v_path}/scripts/base/base.sh"

mkdir -p "${lightx2v_path}/save_results"

torchrun --nproc_per_node="${gpus}" -m lightx2v.infer \
    --model_cls qwen_image \
    --task i2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/nvidia/dist_8/qwen_image_i2i_2511.json" \
    --prompt "Make the girl from Image 1 wear the black dress from Image 2 and sit in the pose from Image 3." \
    --negative_prompt " " \
    --image_path "1.png,2.png,3.png" \
    --save_result_path "${lightx2v_path}/save_results/qwen_image_i2i_2511_dist8.png" \
    --seed 0
