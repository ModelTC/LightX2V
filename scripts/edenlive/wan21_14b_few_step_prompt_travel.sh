#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/EdenLive-x2v

export CUDA_VISIBLE_DEVICES=0
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
source "${lightx2v_path}/scripts/base/base.sh"

# prompt_travel_start_frames contains output video frame indices, starting at 0.
python -m lightx2v.infer \
  --model_cls wan21_edenlive_ar \
  --task i2v \
  --model_path "$model_path" \
  --config_json "${lightx2v_path}/configs/edenlive/wan21_14b_few_step_prompt_travel.json" \
  --image_path /data/nvme6/gushiqiao/codes/vast/EdenLive-Official-wp-demo/examples/samples/cryo-crossing/cover.png \
  --video_path /data/nvme6/gushiqiao/codes/vast/EdenLive-Official-wp-demo/examples/samples/cryo-crossing/recording.mp4 \
  --seed 2026092901 \
  --save_result_path "${lightx2v_path}/save_results/wan21_14b_few_step_prompt_travel.mp4"
