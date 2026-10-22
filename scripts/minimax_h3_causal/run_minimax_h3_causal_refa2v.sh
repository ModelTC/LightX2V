#!/bin/bash
set -eo pipefail

lightx2v_path=/data/nvme6/gushiqiao/codes/report/LightX2V
model_path=/data/nvme6/gushiqiao/models/MiniMax-H3

export CUDA_VISIBLE_DEVICES=5,6

# Set environment variables.
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

# Use minimax_h3_causal_compile.json to enable warmup and torch.compile.
# Warmup runs the configured size/num_frames with a synthetic image and audio.
torchrun --standalone --nproc_per_node=2 -m lightx2v.infer \
  --model_cls minimax_h3_causal \
  --task refa2v \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/minimax_h3_causal/minimax_h3_causal.json" \
  --prompt "The man in the reference image speaks naturally into the microphone, with lip movements synchronized to the audio, subtle facial expressions and natural hand gestures. The camera remains stationary." \
  --image_path "${lightx2v_path}/assets/inputs/audio/seko_input.png" \
  --audio_path "${lightx2v_path}/assets/inputs/audio/seko_input.mp3" \
  --save_result_path "${lightx2v_path}/save_results/h3_refa2v.mp4" \
  --seed 0
