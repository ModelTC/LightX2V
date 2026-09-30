#!/bin/bash

lightx2v_path=
config_json=${lightx2v_path}/configs/fastwam/libero_i2va.json
model_path=

image_path=
state_path=path to state.npy
prompt=

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
