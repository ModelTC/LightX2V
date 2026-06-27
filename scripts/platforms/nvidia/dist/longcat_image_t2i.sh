#!/bin/bash

# LongCat Image T2I 8-GPU Inference Script for NVIDIA
# System management interface: nvidia-smi

lightx2v_path=${LIGHTX2V_PATH:-/data/nvme1/wushuo/LightX2V}
model_path=${MODEL_PATH:-/data/nvme1/wushuo/hf_models/models/LongCat-Image}
gpus=${GPUS:-8}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

source "${lightx2v_path}/scripts/platforms/nvidia/dist/logging.sh"
source "${lightx2v_path}/scripts/base/base.sh"

mkdir -p "${lightx2v_path}/save_results"

torchrun --nproc_per_node="${gpus}" -m lightx2v.infer \
    --model_cls longcat_image \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${lightx2v_path}/configs/platforms/nvidia/dist/longcat_image_t2i_dist8.json" \
    --prompt "一只小猫躺在沙发上" \
    --negative_prompt "" \
    --save_result_path "${lightx2v_path}/save_results/longcat_image_t2i_dist8.png" \
    --seed 42
