#!/bin/bash

# set path firstly
lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

# Keep the original reference image; resample_source_video selects the draft to regenerate.
torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
  --model_cls minimax_h3 \
  --model-variant ref2av \
  --task ref2av \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3_roll/minimax_h3_ref2av_bf16_8step.json" \
  --prompt "The fluffy blue bird from the reference image hops along a wooden garden fence, tilts its head and flaps its small wings. Keep its round body, blue feathers and expressive eyes consistent. A gentle camera pan follows the bird in warm morning sunlight. Sound: cheerful chirping, soft wingbeats and a light breeze." \
  --image_path "${lightx2v_path}/assets/inputs/imgs/flf2v_input_first_frame-fs8.png" \
  --num_frames 124 \
  --size 768 1344 \
  --resample_source_video "${lightx2v_path}/save_results/minimax_h3_roll_ref2av/draft.mp4" \
  --resample_start_step 1 \
  --save_result_path "${lightx2v_path}/save_results/minimax_h3_roll_ref2av/resample.mp4" \
  --seed 42
