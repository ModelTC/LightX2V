#!/bin/bash

# set path firstly
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node=8 -m lightx2v.infer \
--model_cls wan2.1 \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/wan21/wan_t2v_distill_model_4step_cfg_ulysses.json \
--prompt 'Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.' \
--negative_prompt " " \
--save_result_path ${lightx2v_path}/save_results/wan21_t2v_distill_4step.mp4 \
--seed 42
