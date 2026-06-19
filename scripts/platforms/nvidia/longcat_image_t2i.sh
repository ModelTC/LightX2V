#!/bin/bash

# LongCat Image T2I Inference Script
# System management interface: nvidia-smi

lightx2v_path=/data/nvme1/wushuo/LightX2V
model_path=/data/nvme1/wushuo/hf_models/models/LongCat-Image

export CUDA_VISIBLE_DEVICES=0

source "${lightx2v_path}/scripts/platforms/nvidia/logging.sh"
source ${lightx2v_path}/scripts/base/base.sh

# Create output directory
mkdir -p ${lightx2v_path}/save_results

python -m lightx2v.infer \
    --model_cls longcat_image \
    --task t2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/platforms/nvidia/longcat_image_t2i.json \
    --prompt "一只小猫躺在沙发上" \
    --negative_prompt "" \
    --save_result_path ${lightx2v_path}/save_results/longcat_image_t2i.png \
    --seed 42
