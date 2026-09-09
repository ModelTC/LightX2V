#!/bin/bash

# set path firstly
lightx2v_path=/path/to/LightX2V
model_path=/path/to/MiniMax-H3

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

# Ref2AV uses transformer_ref/ weights and supports image, video, and audio references.
# Pass multiple files of one type as comma-separated paths.
# Add --video_path "/path/to/reference.mp4" or --audio_path "/path/to/reference.wav" as needed.
# Audio requires at least one image or video reference.
python -m lightx2v.infer \
  --model_cls minimax_h3 \
  --task ref2av \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3/minimax_h3.json" \
  --prompt "Generate an audio-video scene following the references." \
  --image_path "${lightx2v_path}/assets/inputs/imgs/img_0.jpg" \
  --save_result_path "${lightx2v_path}/save_results/output_lightx2v_minimax_h3_ref2av.mp4" \
  --seed 42
