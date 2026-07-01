#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme5/gushiqiao/codes/LightX2V
model_path=/data/nvme5/gushiqiao/models/Cosmos3-Super
video_path=${model_path}/assets/example_action_id_av_0_input.mp4

export CUDA_VISIBLE_DEVICES=1

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls cosmos3 \
--task v2av \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/cosmos3/cosmos3_super_omni_action_id_av.json \
--prompt "You are an autonomous vehicle planning system." \
--video_path ${video_path} \
--save_result_path ${lightx2v_path}/save_results/cosmos3_super_action_id_av.mp4 \
--save_action_path ${lightx2v_path}/save_results/cosmos3_super_action_id_av.json \
--seed 0
