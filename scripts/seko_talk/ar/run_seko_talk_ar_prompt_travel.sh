#!/bin/bash

lightx2v_path=/data/nvme4/gushiqiao/new/LightX2V
model_path=/data/nvme5/gushiqiao/models/SekoTalk-Distill-AR/

export CUDA_VISIBLE_DEVICES=3

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls seko_talk_ar \
--task rs2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/seko_talk/ar/seko_talk_ar_prompt_travel.json \
--prompt "" \
--negative_prompt "" \
--image_path "/data/nvme4/models/seko_models/0604/20260604-123848.jpg" \
--audio_path "/data/nvme4/models/seko_models/0604/lpm_videos_anna_id2_speak_026_001.mp3" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_seko_talk_ar_prompts.mp4 \
--seed 0
