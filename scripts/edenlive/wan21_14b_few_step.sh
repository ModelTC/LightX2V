#!/bin/bash

lightx2v_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_path="${MODEL_PATH:-${lightx2v_path}/../../vast/EdenLive-x2v}"
: "${IMAGE_PATH:?Set IMAGE_PATH to the input image}"
: "${VIDEO_PATH:?Set VIDEO_PATH to the input video}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "${lightx2v_path}/scripts/base/base.sh"

export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
export PROFILING_DEBUG_LEVEL=0
export OMP_NUM_THREADS=16

python -m lightx2v.infer \
  --model_cls wan21_edenlive_ar \
  --task v2v \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/edenlive/wan21_14b_few_step.json" \
  --prompt 'Cinematic photorealism. A vast volcanic cavern dominated by a churning, incandescent molten lava lake, dark basalt formations, and jagged stalactites. Dynamic heat haze and glowing orange embers fill the air. Seen from a first-person viewpoint, a heavy industrial cryo-blaster with frosted metallic plating and cyan energy vents is held ready. Projectiles flash-freeze bubbling magma into dense slabs of hissing glacial ice. Dramatic lighting balances intense fiery magma radiance against cool cyan cryo luminescence.' \
  --image_path "${IMAGE_PATH}" \
  --video_path "${VIDEO_PATH}" \
  --seed 2026092901 \
  --save_result_path "${lightx2v_path}/save_results/edenlive_14b_few_step.mp4"
