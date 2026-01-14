#!/bin/bash

# set path and first
export lightx2v_path=
export model_path=

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls z_image \
--task i2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/z_image/z_image_turbo_i2i.json \
--prompt "A fantasy landscape with mountains and a river, detailed, vibrant colors" \
--negative_prompt " " \
--image_path ${lightx2v_path}/assets/inputs/imgs/sketch-mountains-input.jpg \
--save_result_path ${lightx2v_path}/save_results/z_image_turbo_i2i.png \
--seed 42 \
--aspect_ratio "1:1"
