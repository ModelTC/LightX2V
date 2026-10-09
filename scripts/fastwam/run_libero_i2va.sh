#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/Wan2.2-TI2V-5B

image_path=/path/to/camera_inputs
state_path=/path/to/state.npy

export CUDA_VISIBLE_DEVICES=6

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
--model_cls fastwam \
--task i2va \
--model_path "${model_path}" \
--config_json "${lightx2v_path}/configs/fastwam/libero_i2va.json" \
--seed 0 \
--prompt "Pick up the black bowl and place it on the plate." \
--image_path "${image_path}" \
--state_path "${state_path}" \
--save_action_path "${lightx2v_path}/save_results/output_fastwam_libero_i2va.actions.npy"
