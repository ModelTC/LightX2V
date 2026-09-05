#!/bin/bash

# set path firstly
lightx2v_path="/data/lightx2v-dev"
model_path="/data/lightx2v-dev/Wan2.1-T2V-14B"

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls wan2.1 \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/meanflow/wan_t2v_meanflow_distill_2step.json \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan_meanflow_t2v.mp4
