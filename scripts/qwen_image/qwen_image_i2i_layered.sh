#!/bin/bash

# set path and first
export lightx2v_path=/path/to/LightX2V
export model_path=/path/to/Qwen/Qwen-Image-Layered
# add the latest diffusers to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/diffusers/src/

export CUDA_VISIBLE_DEVICES=5

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
    --model_cls qwen_image \
    --task i2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/qwen_image/qwen_image_i2i_layered.json \
    --prompt "" \
    --negative_prompt " " \
    --image_path 1.jpeg \
    --save_result_path ${lightx2v_path}/save_results/qwen_image_layered.png \
    --seed 777
