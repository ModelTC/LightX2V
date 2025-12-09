#!/bin/bash

# set path and first
export lightx2v_path=
export model_path=

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
    --model_cls qwen_image \
    --task i2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/qwen_image/qwen_image_i2i.json \
    --prompt "turn the style of the photo to vintage comic book" \
    --negative_prompt " " \
    --image_path pie.png \
    --save_result_path ${lightx2v_path}/save_results/qwen_image_i2i.png \
    --seed 0
