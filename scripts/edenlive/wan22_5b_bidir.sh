#!/bin/bash

lightx2v_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_path="${MODEL_PATH:-${lightx2v_path}/../../vast/EdenLive-x2v}"
: "${IMAGE_PATH:?Set IMAGE_PATH to the cover image}"
: "${VIDEO_PATH:?Set VIDEO_PATH to the recording}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "${lightx2v_path}/scripts/base/base.sh"

export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
export PROFILING_DEBUG_LEVEL=0
export OMP_NUM_THREADS=16

python -m lightx2v.infer \
  --model_cls wan22_edenlive \
  --task v2v \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/edenlive/wan22_5b_bidir.json" \
  --prompt "A cinematic scene with detailed, realistic lighting." \
  --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多" \
  --image_path "${IMAGE_PATH}" \
  --video_path "${VIDEO_PATH}" \
  --seed 42 \
  --save_result_path "${lightx2v_path}/save_results/edenlive_5b_bidir.mp4"
