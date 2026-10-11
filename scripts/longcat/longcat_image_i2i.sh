#!/bin/bash

# LongCat Image Edit (I2I) Inference Script
# Usage: bash longcat_image_i2i.sh

lightx2v_path=/path/to/LightX2V
model_path=/path/to/LongCat-Image-Edit
export CUDA_VISIBLE_DEVICES=0

source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
    --model_cls longcat_image \
    --task i2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/longcat_image/longcat_image_i2i.json \
    --prompt "将猫变成狗" \
    --negative_prompt "" \
    --image_path ${model_path}/assets/test.png \
    --save_result_path ${lightx2v_path}/save_results/longcat_image_i2i.png \
    --seed 43
