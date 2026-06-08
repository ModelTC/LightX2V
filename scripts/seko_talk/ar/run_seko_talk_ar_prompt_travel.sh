#!/bin/bash

lightx2v_path=/data/nvme4/gushiqiao/new/seko/LightX2V
model_path=/data/nvme5/gushiqiao/models/SekoTalk-Distill-AR/

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls seko_talk_ar \
--task rs2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/seko_talk/ar/seko_talk_ar_prompt_travel_4gpu.json \
--prompt "" \
--negative_prompt "" \
--image_path "/data/nvme4/models/seko_models/0604/20260604-123848.jpg" \
--audio_path "/data/nvme4/gushiqiao/new/seko/LightX2V/save_results/1.wav" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_seko_talk_ar_prompts.mp4 \
--seed 0
