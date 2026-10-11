#!/bin/bash

# LongCat Image T2I Inference with CPU Offload
# Usage: bash longcat_image_t2i_offload.sh

lightx2v_path=/path/to/LightX2V
model_path=/path/to/LongCat-Image
export CUDA_VISIBLE_DEVICES=0

source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
    --model_cls longcat_image \
    --task t2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/longcat_image/longcat_image_t2i_offload.json \
    --prompt "一只小猫躺在沙发上" \
    --negative_prompt "" \
    --save_result_path ${lightx2v_path}/save_results/longcat_image_t2i_offload.png \
    --seed 42
