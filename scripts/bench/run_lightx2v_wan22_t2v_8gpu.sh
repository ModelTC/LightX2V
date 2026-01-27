#!/bin/bash

set -x
# set path and first
lightx2v_path=

model_path=/path/to/Wan-AI/Wan2.2-T2V-A14B

config_json=${lightx2v_path}/configs/dist_infer/wan22_moe_t2v_cfg_ulysses.json

model_cls=wan2.2_moe

task=t2v

prompt="A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."

negative_prompt=" "

gpus=8
ts=$(date +"%y%m%d%H%M%S")

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node=${gpus} -m lightx2v.infer \
--target_shape 720 1280 \
--seed 42 \
--model_cls ${model_cls} \
--task ${task} \
--model_path $model_path \
--config_json ${config_json} \
--prompt "${prompt}" \
--negative_prompt "${negative_prompt}" \
--save_result_path ${lightx2v_path}/save_results/${model_cls}_${task}_gpu${gpus}_${ts}.mp4
