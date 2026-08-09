#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme6/gushiqiao/codes/LightX2V
model_path=/data/nvme6/gushiqiao/models/MiniMax-H3

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

python -m lightx2v.infer \
--model_cls minimax_h3 \
--task fl2av \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/minimax_h3/minimax_h3_fl2av.json \
--prompt "Create a coherent transition with natural synchronized sound." \
--image_path ${lightx2v_path}/assets/inputs/imgs/flf2v_input_first_frame-fs8.png \
--last_frame_path ${lightx2v_path}/assets/inputs/imgs/flf2v_input_last_frame-fs8.png \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_minimax_h3_fl2av.mp4 \
--seed 42
