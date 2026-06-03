#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/wushuo/LightX2V
hidream_o1_image_path=/data/nvme1/wushuo/HiDream-O1-Image
model_path=/data/nvme1/wushuo/hf_models/HiDream-O1-Image

export CUDA_VISIBLE_DEVICES=0

# keep the same effective inputs/outputs as HiDream-O1-Image/hidream_o1_image_i2i_multi.sh
prompt="A young boy with blonde hair stands on steps wearing light blue jeans, a white t-shirt with logo, and blue and white sneakers. He wears a brown cord necklace with beads, a black wristwatch with digital display, and carries a yellow fanny pack with white zipper. In his hand is a red boxing glove with white top, a teal plastic toy car, and a plastic toy figure of Captain America. He wears a straw hat with cream band. Natural light illuminates the scene."
ref_images="${hidream_o1_image_path}/assets/IP/1.jpg,${hidream_o1_image_path}/assets/IP/2.jpg,${hidream_o1_image_path}/assets/IP/3.jpg,${hidream_o1_image_path}/assets/IP/4.jpg,${hidream_o1_image_path}/assets/IP/5.jpg,${hidream_o1_image_path}/assets/IP/6.jpg,${hidream_o1_image_path}/assets/IP/7.jpg,${hidream_o1_image_path}/assets/IP/8.jpg,${hidream_o1_image_path}/assets/IP/9.jpg,${hidream_o1_image_path}/assets/IP/10.jpg"
output_image=${hidream_o1_image_path}/results/subject.png

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls hidream_o1_image \
--task i2i \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/hidream_o1_image/hidream_o1_image_i2i_multi.json \
--prompt "${prompt}" \
--image_path "${ref_images}" \
--save_result_path "${output_image}" \
--seed 32
