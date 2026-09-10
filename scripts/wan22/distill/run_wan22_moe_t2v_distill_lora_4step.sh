#!/bin/bash

# set path firstly
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls wan2.2_moe \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/distill/wan22/wan_moe_t2v_distill_lora.json \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan22_moe_t2v_distill.mp4
