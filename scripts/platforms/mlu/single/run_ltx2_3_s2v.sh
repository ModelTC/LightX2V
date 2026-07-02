#!/bin/bash

# set path firstly
lightx2v_path=/data/LightX2V
model_path=/data/models/LTX-2
AUDIO_PATH=/data/ltx2_s2v_sample.wav

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES=0
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:${LD_LIBRARY_PATH}

source "${lightx2v_path}/scripts/platforms/mlu/single/logging.sh"
source ${lightx2v_path}/scripts/platforms/mlu/logging.sh

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"

mkdir -p "${lightx2v_path}/save_results"

python -m lightx2v.infer \
  --model_cls ltx2 \
  --task ltx2_s2v \
  --model_path "${model_path}" \
  --config_json ${lightx2v_path}/configs/platforms/mlu/single/ltx2_3.json \
  --audio_path "${AUDIO_PATH}" \
  --prompt "A person speaks clearly in a quiet room, natural lighting, cinematic medium shot." \
  --negative_prompt "blurry, out of focus, overexposed, underexposed, low contrast, excessive noise, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, disfigured hands, artifacts, inconsistent perspective, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, off-sync audio, AI artifacts." \
  --save_result_path "${lightx2v_path}/save_results/output_lightx2v_ltx2_s2v_mlu.mp4"
