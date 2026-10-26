#!/bin/bash

# set path firstly
lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

# This example uses a K3 draft; remove --draft_steps to use the full configured schedule.
torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
  --model_cls minimax_h3 \
  --model-variant fl2av \
  --task fl2av \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3_roll/minimax_h3_bf16_8step.json" \
  --prompt "A fluffy blue bird stands on a wooden ledge, looks up, spreads its wings and takes off into a bright blue sky. The camera follows the bird upward in one continuous shot. Sound: soft chirping, fluttering wings and a gentle breeze." \
  --image_path "${lightx2v_path}/assets/inputs/imgs/flf2v_input_first_frame-fs8.png" \
  --last_frame_path "${lightx2v_path}/assets/inputs/imgs/flf2v_input_last_frame-fs8.png" \
  --num_frames 124 \
  --size 480 864 \
  --draft_steps 3 \
  --save_result_path "${lightx2v_path}/save_results/minimax_h3_roll_fl2av/draft.mp4" \
  --seed 42
