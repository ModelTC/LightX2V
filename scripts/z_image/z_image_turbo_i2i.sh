#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/yongyang/nb/LightX2V
model_path=/data/nvme1/models/Tongyi-MAI/Z-Image-Turbo
image_path=${lightx2v_path}/assets/inputs/imgs/img_0.jpg

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls z_image \
--task i2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/z_image/z_image_turbo_i2i.json \
--image_path $image_path \
--prompt "Change the cat to a dog." \
--negative_prompt " " \
--save_result_path ${lightx2v_path}/save_results/z_image_turbo_i2i.png \
--seed 42 \
--i2i_denoise_strength 1.0
