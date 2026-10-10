#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/SekoTalk-v2.7_beta2-fp8-step4

export CUDA_VISIBLE_DEVICES=6

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh


python -m lightx2v.infer \
--model_cls seko_talk \
--task s2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/seko_talk/seko_talk_02_fp8.json \
--prompt  "让角色根据音频内容自然说话" \
--image_path /path/to/input.png \
--audio_path ${lightx2v_path}/assets/inputs/audio/seko_input.mp3 \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_seko_talk.mp4
