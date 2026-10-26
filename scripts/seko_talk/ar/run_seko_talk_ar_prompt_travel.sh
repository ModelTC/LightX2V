#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/SekoTalk-Distill-AR

export CUDA_VISIBLE_DEVICES=3

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls seko_talk_ar \
--task rs2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/seko_talk/ar/seko_talk_ar_prompt_travel.json \
--image_path /path/to/input.png \
--audio_path /path/to/input.mp3 \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_seko_talk_ar_prompts.mp4 \
--seed 0
