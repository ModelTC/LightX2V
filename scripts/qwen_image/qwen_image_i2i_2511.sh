#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/yongyang/ddc/yong/LightX2V
model_path=/data/nvme1/models/qwen-image-edit-release-251130

export CUDA_VISIBLE_DEVICES=5

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
    --model_cls qwen_image \
    --task i2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/qwen_image/qwen_image_i2i_2511.json \
    --prompt "Change the person to a standing position, bending over to hold the dog's front paws." \
    --negative_prompt " " \
    --image_path "/data/nvme1/yongyang/ddc/lightx2v_examples/i2i/1/img1.png" \
    --save_result_path ${lightx2v_path}/save_results/qwen_image_i2i_2511.png \
    --seed 0
