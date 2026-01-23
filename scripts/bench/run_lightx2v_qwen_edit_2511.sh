#!/bin/bash

set -x
# set path and first
lightx2v_path=

model_path=/path/to/Qwen/Qwen-Image-Edit-2511

image_path=${lightx2v_path}/assets/inputs/imgs/cat.png

config_json=${lightx2v_path}/configs/qwen_image/qwen_image_i2i_2511.json

model_cls=qwen_image

task=i2i

prompt="Transform into anime style"

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
--image_path ${image_path} \
--save_result_path ${lightx2v_path}/save_results/${model_cls}_${task}_gpu${gpus}_${ts}.png
