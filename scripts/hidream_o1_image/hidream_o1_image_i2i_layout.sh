#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/wushuo/LightX2V
hidream_o1_image_path=/data/nvme1/wushuo/HiDream-O1-Image
model_path=/data/nvme1/wushuo/hf_models/HiDream-O1-Image

export CUDA_VISIBLE_DEVICES=0

# keep the same effective inputs/outputs as HiDream-O1-Image/hidream_o1_image_i2i_layout.sh
prompt="City council members pose with relaxed smiles on a sunlit terrace, warm approachable mood, golden hour, cinematic soft glow."
ref_images="${hidream_o1_image_path}/assets/IP_layout/0.jpg,${hidream_o1_image_path}/assets/IP_layout/1.jpg"
layout_bboxes="[[0.20507812, 0.43945312, 0.48828125, 0.7421875 ], [0.57617188, 0.80078125, 0.08789062, 0.34179688]]"
output_image=${hidream_o1_image_path}/results/ip_layout.png

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls hidream_o1_image \
--task i2i \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/hidream_o1_image/hidream_o1_image_i2i_layout.json \
--prompt "${prompt}" \
--image_path "${ref_images}" \
--layout_bboxes "${layout_bboxes}" \
--save_result_path "${output_image}" \
--seed 42
