#!/bin/bash
set -e

# set path firstly
lightx2v_path=/path/to/LightX2V
model_path=/path/to/TBDub
video_path=/path/to/cropped_25fps.mp4
audio_path=/path/to/driving.wav

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
--model_cls tbdub \
--task v2v \
--model_path "${model_path}" \
--config_json "${lightx2v_path}/configs/wan22/tbdub.json" \
--video_path "${video_path}" \
--audio_path "${audio_path}" \
--seed 42 \
--save_result_path "${lightx2v_path}/save_results/output_lightx2v_tbdub_v2v.mp4"
