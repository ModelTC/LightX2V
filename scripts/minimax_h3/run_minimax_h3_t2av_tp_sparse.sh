#!/bin/bash

lightx2v_path=/data/nvme6/gushiqiao/codes/LightX2V
model_path=/data/wushuo1/models/wyr_models/minimax_h3

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

source ${lightx2v_path}/scripts/base/base.sh
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

prompt='In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.'

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
--model_cls minimax_h3 \
--task t2av \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/minimax_h3/minimax_h3_t2av_tp.json \
--prompt "$prompt" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av_tp.mp4 \
--seed 0 \
--warmup > ${lightx2v_path}/save_results/minimax_h3_t2av_544p_124_8gpu_tp8.log 2>&1 &
