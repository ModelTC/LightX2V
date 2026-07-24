#!/bin/bash

set -eo pipefail

lightx2v_path=/data/LightX2V
model_path=/data/models/LTX-2
audio_path="${lightx2v_path}/assets/inputs/audio/seko_input.mp3"

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-0}"
export PYTORCH_MLU_ALLOC_CONF="${PYTORCH_MLU_ALLOC_CONF:-expandable_segments:True}"
export LD_LIBRARY_PATH="/usr/local/neuware/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
  --model_cls ltx2 \
  --task ltx2_s2v \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/platforms/mlu/ltx2_3.json" \
  --audio_path "${audio_path}" \
  --prompt "A person speaks clearly in a quiet room, natural lighting, cinematic medium shot." \
  --negative_prompt "blurry, out of focus, overexposed, underexposed, low contrast, excessive noise, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, disfigured hands, artifacts, inconsistent perspective, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, off-sync audio, AI artifacts." \
  --save_result_path "${lightx2v_path}/save_results/output_lightx2v_ltx2_3_s2v_mlu.mp4" \
  --seed 42
