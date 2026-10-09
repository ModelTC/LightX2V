#!/bin/bash
set -e

lightx2v_path=""
config_json="${lightx2v_path}/configs/realtimewam/libero_fasterwam_i2va.json"
model_path=""

image_path=""
state_path=""
prompt=""
save_action_path=""

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
--model_cls realtimewam \
--task i2va \
--model_path "${model_path}" \
--config_json "${config_json}" \
--seed 0 \
--prompt "${prompt}" \
--image_path "${image_path}" \
--state_path "${state_path}" \
--save_action_path "${save_action_path}"
