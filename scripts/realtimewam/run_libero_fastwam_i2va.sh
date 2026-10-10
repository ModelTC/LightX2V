#!/bin/bash
set -e

lightx2v_path=/path/to/LightX2V
model_path=/path/to/Wan2.2-TI2V-5B

image_path=${lightx2v_path}/examples/realtimewam/assets
state_path=${image_path}/state.npy

export CUDA_VISIBLE_DEVICES=0

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
--model_cls realtimewam \
--task i2va \
--model_path "${model_path}" \
--config_json "${lightx2v_path}/configs/realtimewam/libero_fastwam_i2va.json" \
--seed 0 \
--prompt "pick up the black bowl between the plate and the ramekin and place it on the plate" \
--image_path "${image_path}" \
--state_path "${state_path}" \
--save_action_path "${lightx2v_path}/save_results/realtimewam_libero_fast.actions.npy"
