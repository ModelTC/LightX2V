#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/yongyang/nb/LightX2V
model_path=/data/nvme1/yongyang/nb/models/GEAR-Dreams/DreamZero-DROID
input_path=/data/nvme1/yongyang/nb/dreamzero/debug_image

export CUDA_VISIBLE_DEVICES=5

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls dreamzero \
--task i2va \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/dreamzero/dreamzero_droid_i2va.json \
--seed 1140 \
--prompt "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan" \
--negative_prompt "Vibrant colors, overexposed, static, blurry details, text, subtitles, style, artwork, painting, image, still, grayscale, dull, worst quality, low quality, JPEG artifacts, ugly, mutilated, extra fingers, bad hands, bad face, deformed, disfigured, mutated limbs, fused fingers, stagnant image, cluttered background, three legs, many people in the background, walking backwards." \
--image_path $input_path \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_dreamzero_droid_i2va.mp4 \
--save_action_path ${lightx2v_path}/save_results/output_lightx2v_dreamzero_droid_i2va_actions.npy
