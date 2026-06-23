#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/yongyang/nb/LightX2V
model_path=/data/nvme1/yongyang/nb/models/HiDream-ai/HiDream-O1-Image

export CUDA_VISIBLE_DEVICES=0

# keep the same effective inputs/outputs as HiDream-O1-Image/hidream_o1_image_i2i.sh
prompt="remove the earphones"
ref_images=/data/nvme1/yongyang/nb/HiDream-O1-Image/assets/edit/test.jpg
output_image=${lightx2v_path}/results/edit.png

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls hidream_o1_image \
--task i2i \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/hidream_o1_image/hidream_o1_image_i2i.json \
--prompt "${prompt}" \
--image_path "${ref_images}" \
--i2i_denoise_strength 0.6 \
--save_result_path "${output_image}" \
--seed 32
