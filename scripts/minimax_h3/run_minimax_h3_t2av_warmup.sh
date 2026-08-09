#!/bin/bash

# set path firstly
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

prompt='A cinematic fox walking through a snowy forest.'

python -m lightx2v.infer \
--model_cls minimax_h3 \
--task t2av \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/minimax_h3/minimax_h3_t2av_compile.json \
--warmup \
--prompt "$prompt" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av10.mp4 \
--seed 0
