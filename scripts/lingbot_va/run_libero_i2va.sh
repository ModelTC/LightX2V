#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme4/gushiqiao/new/LightX2V
model_path=/data/nvme5/gushiqiao/models/lingbot-va-posttrain-libero-long/

export CUDA_VISIBLE_DEVICES=1

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls lingbot_va \
--task i2va \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/lingbot_va/libero_i2va.json \
--prompt "put both the alphabet soup and the tomato sauce in the basket" \
--negative_prompt "" \
--image_path /data/nvme4/gushiqiao/new/lingbot-va/example/libero \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_lingbot_va_libero_i2va.mp4
