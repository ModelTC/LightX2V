#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme5/gushiqiao/codes/LightX2V
model_path=/data/nvme5/gushiqiao/models/Cosmos3-Super
image_path=${model_path}/assets/example_action_fd_agibotworld_first_frame.png
action_path=${model_path}/assets/example_action_fd_agibotworld_action_chunks.json

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls cosmos3 \
--task i2va \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/cosmos3/cosmos3_super_omni_action_fd_agibotworld.json \
--prompt "" \
--image_path ${image_path} \
--action_path ${action_path} \
--save_result_path ${lightx2v_path}/save_results/cosmos3_super_action_fd_agibotworld.mp4 \
--seed 0
