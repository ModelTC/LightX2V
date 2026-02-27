#!/bin/bash

# set path firstly
lightx2v_path=
model_path= # path to Wan2.1-T2V-1.3B

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls wan2.1_sf \
--task t2v \
--model_path $model_path \
--sf_model_path $sf_model_path \
--config_json ${lightx2v_path}/configs/self_forcing/wan_t2v_sf.json \
--prompt 'Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.' \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan_t2v_sf_nvfp4.mp4
