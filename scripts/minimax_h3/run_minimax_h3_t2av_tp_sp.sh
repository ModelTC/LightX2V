#!/bin/bash

lightx2v_path=/data/nvme6/gushiqiao/codes/LightX2V
model_path=/data/nvme6/gushiqiao/models/MiniMax-H3

export CUDA_VISIBLE_DEVICES=0,1,2,3

source ${lightx2v_path}/scripts/base/base.sh
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32

torchrun --standalone --nproc_per_node=4 -m lightx2v.infer \
--model_cls minimax_h3 \
--task t2av \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/minimax_h3/minimax_h3_t2av_tp_sp.json \
--prompt "A cinematic fox walking through a snowy forest" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av_tp_sp.mp4 \
--seed 42
