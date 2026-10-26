#!/usr/bin/env bash
set -euo pipefail

lightx2v_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_path="${MODEL_PATH:-${lightx2v_path}/../models/H3/MiniMax-H3}"
config_path="${CONFIG_PATH:-${lightx2v_path}/configs/minimax_h3_causal/realtime_768p_bf16_5090_sp8.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Set environment variables.
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
  --model_cls minimax_h3_causal \
  --task refa2v \
  --model_path "${model_path}" \
  --config_json "${config_path}" \
  --prompt "The man in the reference image speaks naturally into the microphone, with lip movements synchronized to the audio, subtle facial expressions and natural hand gestures. The camera remains stationary." \
  --image_path "${IMAGE_PATH:-${lightx2v_path}/assets/inputs/audio/seko_input.png}" \
  --audio_path "${AUDIO_PATH:-${lightx2v_path}/assets/inputs/audio/seko_input.mp3}" \
  --save_result_path "${OUTPUT_PATH:-${lightx2v_path}/save_results/h3_refa2v.mp4}" \
  --seed "${SEED:-0}"
