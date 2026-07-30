#!/bin/bash

# set path firstly
lightx2v_path=/app/LightX2V
model_path=/app/nvidia_cosmos3_models/Cosmos3-Nano-Policy-DROID
input_path=/app/lightx2v_examples/i2va/robolab/banana_in_bowl
image_path=${input_path}/observation.png
state_path=${input_path}/state.npy

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node=8 -m lightx2v.infer \
--model_cls cosmos3 \
--task i2va \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/cosmos3/cosmos3_nano_policy_droid_cfg_ulysses_8gpu.json \
--prompt "Pick up the banana and place it in the bowl" \
--image_path ${image_path} \
--state_path ${state_path} \
--save_action_path ${lightx2v_path}/save_results/cosmos3_nano_policy_droid_action.npy \
--seed 0
