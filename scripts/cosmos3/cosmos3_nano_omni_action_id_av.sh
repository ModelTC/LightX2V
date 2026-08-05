#!/bin/bash

# set path firstly
lightx2v_path=/root/yongyang/LightX2V
model_path=/root/yongyang/models/nvidia_cosmos3_models/Cosmos3-Nano
video_path=${model_path}/assets/example_action_id_av_0_input.mp4

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls cosmos3 \
--task v2av \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/cosmos3/cosmos3_nano_omni_action_id_av.json \
--prompt "You are an autonomous vehicle planning system." \
--video_path ${video_path} \
--save_result_path ${lightx2v_path}/save_results/cosmos3_nano_action_id_av.mp4 \
--save_action_path ${lightx2v_path}/save_results/cosmos3_nano_action_id_av.json \
--seed 0
