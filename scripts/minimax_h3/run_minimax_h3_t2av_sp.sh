#!/bin/bash

# set path firstly
lightx2v_path=/path/to/LightX2V
model_path=/path/to/models/minimax_h3/h3_hf_bf16

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
--model_cls minimax_h3 \
--task t2av \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/minimax_h3/fp8/minimax_h3_t2av_sp8_5090.json \
--prompt "A cinematic fox walking through a snowy forest" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av_sp.mp4 \
--seed 42
