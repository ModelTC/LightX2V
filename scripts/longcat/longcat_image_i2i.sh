#!/bin/bash

# LongCat Image Edit (I2I) Inference Script
# Usage: bash longcat_image_i2i.sh

export lightx2v_path=/workspace
export model_path=/workspace/models/LongCat-Image-Edit
export CUDA_VISIBLE_DEVICES=0

# Source base configuration if exists
if [ -f "${lightx2v_path}/scripts/base/base.sh" ]; then
    source ${lightx2v_path}/scripts/base/base.sh
fi

# Create output directory
mkdir -p ${lightx2v_path}/save_results

python -m lightx2v.infer \
    --model_cls longcat_image \
    --task i2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/longcat_image/longcat_image_i2i.json \
    --prompt "将猫变成狗" \
    --negative_prompt "" \
    --image_path ${lightx2v_path}/models/LongCat-Image-Edit/assets/test.png \
    --save_result_path ${lightx2v_path}/save_results/longcat_image_i2i.png \
    --seed 43
