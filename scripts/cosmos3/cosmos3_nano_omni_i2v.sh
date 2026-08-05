#!/bin/bash

# set path firstly
lightx2v_path=/root/yongyang/LightX2V
model_path=/root/yongyang/models/nvidia_cosmos3_models/Cosmos3-Nano
prompt_path=${model_path}/assets/example_i2v_prompt.json
negative_prompt_path=${model_path}/assets/negative_prompt.json
image_path=${model_path}/assets/example_i2v_input.jpg

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls cosmos3 \
--task i2v \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/cosmos3/cosmos3_nano_omni_i2v.json \
--prompt ${prompt_path} \
--negative_prompt ${negative_prompt_path} \
--image_path ${image_path} \
--save_result_path ${lightx2v_path}/save_results/cosmos3_nano_omni_i2v.mp4 \
--seed 17
