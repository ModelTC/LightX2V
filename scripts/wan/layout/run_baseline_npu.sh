#!/bin/bash

lightx2v_path="$(cd "$(dirname "$0")/../../.." && pwd)"
model_path="${MODEL_PATH:-${lightx2v_path}/models/Wan2.1-I2V-14B-720P}"

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
  --model_cls wan2.1 \
  --task i2v \
  --model_path "${model_path}" \
  --config_json "${CONFIG_PATH:-${lightx2v_path}/configs/wan/layout/baseline_npu.json}" \
  --prompt "A white cat wearing sunglasses sits on a surfboard on a sunny beach." \
  --negative_prompt "过曝，静态，模糊，字幕，低质量，畸形" \
  --image_path "${lightx2v_path}/assets/inputs/imgs/img_0.jpg" \
  --save_result_path "${lightx2v_path}/save_results/wan_layout/baseline_npu.mp4" \
  --seed 42
