#!/bin/bash

# set path firstly
lightx2v_path=/path/to/LightX2V
# Native 2x Wan + Flash inference checkpoints
model_path=/path/to/weights_relayvsr_x2
video_path=/path/to/input.mp4

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
  --model_cls relayvsr_x2 \
  --task sr \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/relayvsr/relayvsr_x2.json" \
  --video_path "${video_path}" \
  --sr_ratio 2 \
  --save_result_path "${lightx2v_path}/save_results/relayvsr_x2.mp4" \
  --seed 42
