#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/SekoTalk-Distill/

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh


python -m lightx2v.infer \
--model_cls seko_talk \
--task s2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/seko_talk/5090/seko_talk_5090_bf16.json \
--prompt  "The video features a male speaking to the camera with arms spread out, a slightly furrowed brow, and a focused gaze." \
--image_path ${lightx2v_path}/assets/inputs/audio/seko_input.png \
--audio_path ${lightx2v_path}/assets/inputs/audio/seko_input.mp3 \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_seko_talk.mp4
