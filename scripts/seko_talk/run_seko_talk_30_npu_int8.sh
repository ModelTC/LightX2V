#!/bin/bash

lightx2v_path=/data/wq/proj/sd/code/LightX2V
model_path=/root/SekoTalk-Distill-int8
export ASCEND_RT_VISIBLE_DEVICES=4
export PLATFORM=ascend_npu
export DTYPE="BF16"

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh


python -m lightx2v.infer \
--model_cls seko_talk \
--task s2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/seko_talk/npu/seko_talk_02_int8.json \
--prompt  "The video features a male speaking to the camera with arms spread out, a slightly furrowed brow, and a focused gaze." \
--image_path ${lightx2v_path}/assets/inputs/audio/seko_input.png \
--audio_path ${lightx2v_path}/assets/inputs/audio/seko_input.mp3 \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_seko_talk_npu_int8.mp4
