#!/bin/bash

# set path and first
export lightx2v_path=
export model_path=

export CUDA_VISIBLE_DEVICES=

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
    --model_cls qwen_image \
    --task i2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/offload/block/qwen_image_i2i_2509_block.json \
    --prompt "Have the two characters swap clothes and stand in front of the castle." \
    --negative_prompt " " \
    --image_path 1.jpeg,2.jpeg \
    --save_result_path ${lightx2v_path}/save_results/qwen_image_i2i_2509.png \
    --seed 0
