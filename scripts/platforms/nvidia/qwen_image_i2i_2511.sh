#!/bin/bash

# System management interface: nvidia-smi

# set path and first
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
    --model_cls qwen_image \
    --task i2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/platforms/nvidia/qwen_image_i2i_2511.json \
    --prompt "Make the girl from Image 1 wear the black dress from Image 2 and sit in the pose from Image 3." \
    --negative_prompt " " \
    --image_path "1.png,2.png,3.png" \
    --save_result_path ${lightx2v_path}/save_results/qwen_image_i2i_2511.png \
    --seed 0
