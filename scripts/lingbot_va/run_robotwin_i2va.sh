#!/bin/bash

# set path firstly
lightx2v_path=/path/to/LightX2V
model_path=/path/to/lingbot-va-posttrain-robotwin

export CUDA_VISIBLE_DEVICES=1

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls lingbot_va \
--task i2va \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/lingbot_va/robotwin_i2va.json \
--prompt "Grab the medium-sized white mug, rotate it, place it on the table, and hook it onto the smooth dark gray rack." \
--negative_prompt "" \
--image_path /path/to/lingbot-va/example/robotwin \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_lingbot_va_robotwin_i2va.mp4
