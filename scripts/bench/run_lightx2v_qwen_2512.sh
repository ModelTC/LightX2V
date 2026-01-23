#!/bin/bash

set -x
# set path and first
lightx2v_path=

model_path=/path/to/Qwen/Qwen-Image-2512

config_json=${lightx2v_path}/configs/qwen_image/qwen_image_t2i_2512.json

model_cls=qwen_image

task=t2i

prompt="A futuristic cyberpunk city at night, neon lights reflecting on wet streets, highly detailed, 8k"

negative_prompt=" "

gpus=1
ts=$(date +"%y%m%d%H%M%S")

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--target_shape 1024 1024 \
--seed 42 \
--model_cls ${model_cls} \
--task ${task} \
--model_path $model_path \
--config_json ${config_json} \
--prompt "${prompt}" \
--negative_prompt "${negative_prompt}" \
--save_result_path ${lightx2v_path}/save_results/${model_cls}_${task}_gpu${gpus}_${ts}.png
