#!/bin/bash

lightx2v_path=/data/nvme7/yongyang/LightX2V
config_json=${lightx2v_path}/configs/fastwam/libero_i2va.json
model_path=/data/nvme7/yongyang/models/Wan-AI/Wan2.2-TI2V-5B

image_path=/data/nvme7/yongyang/lightx2v_examples/i2va/libero_spatial/task0_init0
state_path=/data/nvme7/yongyang/lightx2v_examples/i2va/libero_spatial/task0_init0/state.npy
prompt="pick up the black bowl between the plate and the ramekin and place it on the plate"

export CUDA_VISIBLE_DEVICES=6

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
--model_cls fastwam \
--task i2va \
--model_path "${model_path}" \
--config_json "${config_json}" \
--seed 0 \
--prompt "${prompt}" \
--image_path "${image_path}" \
--state_path "${state_path}" \
--save_action_path "${lightx2v_path}/save_results/output_fastwam_libero_i2va.actions.npy"
