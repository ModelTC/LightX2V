#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/EdenLive-x2v

export CUDA_VISIBLE_DEVICES=0
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
  --model_cls wan21_edenlive_ar \
  --task i2v \
  --model_path "$model_path" \
  --config_json "${lightx2v_path}/configs/edenlive/offload/wan21_14b_few_step_block.json" \
  --prompt 'A first-person view of a volcanic cavern. A cryo-blaster freezes lava into ice platforms.' \
  --image_path /absolute/path/to/cover.png \
  --video_path /absolute/path/to/recording.mp4 \
  --seed 2026092901 \
  --save_result_path "${lightx2v_path}/save_results/edenlive_block_offload.mp4"
