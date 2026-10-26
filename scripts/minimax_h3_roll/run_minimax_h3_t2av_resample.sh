#!/bin/bash

# set path firstly
lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
  --model_cls minimax_h3 \
  --model-variant fl2av \
  --task t2av \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3_roll/minimax_h3_bf16_8step.json" \
  --prompt "A yellow tram passes along a rainy street. Sound: rainfall and a tram bell." \
  --num_frames 124 \
  --size 768 1344 \
  --resample_source_video "${lightx2v_path}/save_results/minimax_h3_roll_t2av/draft.mp4" \
  --resample_start_step 1 \
  --save_result_path "${lightx2v_path}/save_results/minimax_h3_roll_t2av/resample.mp4" \
  --seed 42
