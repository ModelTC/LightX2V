#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/wushuo/LightX2V
hidream_o1_image_path=/data/nvme1/wushuo/HiDream-O1-Image
model_path=/data/nvme1/wushuo/hf_models/HiDream-O1-Image

export CUDA_VISIBLE_DEVICES=2

# keep the same effective inputs/outputs as HiDream-O1-Image/hidream_o1_image_i2i.sh
prompt="remove the earphones"
ref_images=${hidream_o1_image_path}/assets/edit/test.jpg
output_image=${hidream_o1_image_path}/results/edit.png

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls hidream_o1_image \
--task i2i \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/hidream_o1_image/hidream_o1_image_i2i.json \
--prompt "${prompt}" \
--image_path "${ref_images}" \
--save_result_path "${output_image}" \
--seed 32
