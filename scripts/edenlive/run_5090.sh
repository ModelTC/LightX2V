#!/usr/bin/env bash
set -euo pipefail

lightx2v_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_path="${MODEL_PATH:-${lightx2v_path}/../../vast/EdenLive-x2v}"
config_path="${CONFIG_PATH:-${lightx2v_path}/configs/edenlive/realtime/wan22_5b_ar_fp8_f16_accum_sage_lighttae_async_5090.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "${lightx2v_path}/scripts/base/base.sh"

export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
export PROFILING_DEBUG_LEVEL=0
export OMP_NUM_THREADS=16

# Set quantized checkpoint and LightTAE paths in the JSON below.

python -m lightx2v.infer \
  --model_cls wan22_edenlive_ar \
  --task v2v \
  --model_path "${model_path}" \
  --config_json "${config_path}" \
  --prompt "${PROMPT:-A cinematic scene with detailed, realistic lighting.}" \
  --image_path "${IMAGE_PATH:?Set IMAGE_PATH to the cover image}" \
  --video_path "${VIDEO_PATH:?Set VIDEO_PATH to the recording}" \
  --seed "${SEED:-2026092901}" \
  --save_result_path "${OUTPUT_PATH:-${lightx2v_path}/save_results/edenlive_5b_5090.mp4}"
