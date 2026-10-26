#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/EdenLive-x2v

export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
export PROFILING_DEBUG_LEVEL=0
export OMP_NUM_THREADS=16

# Set quantized checkpoint and LightTAE paths in the JSON below.

python -m lightx2v.infer \
  --model_cls wan22_edenlive_ar \
  --task i2v \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/edenlive/realtime/wan22_5b_ar_fp8_sgl_hopper.json" \
  --prompt "A cinematic scene with detailed, realistic lighting." \
  --image_path /path/to/cover.png \
  --video_path /path/to/recording.mp4 \
  --seed 2026092901 \
  --save_result_path "${lightx2v_path}/save_results/edenlive_5b_hopper_fp8_sgl_24fps.mp4"
