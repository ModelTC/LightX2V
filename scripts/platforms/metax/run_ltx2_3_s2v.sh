#!/bin/bash

# LTX-2.3 S2V Inference Script for MetaX
# System management interface: mx-smi

lightx2v_path=/data/LightX2V
model_path=/data/models/LTX-2
AUDIO_PATH=/data/ltx2_s2v_sample.wav

export PLATFORM="metax_cuda"
export CUDA_VISIBLE_DEVICES=0
export MACA_PATH=${MACA_PATH:-/opt/maca-3.3.0}
export PATH=${MACA_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${LD_LIBRARY_PATH}

source ${lightx2v_path}/scripts/platforms/metax/logging.sh

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
  --model_cls ltx2 \
  --task ltx2_s2v \
  --model_path "${model_path}" \
  --config_json ${lightx2v_path}/configs/platforms/metax/ltx2_3.json \
  --audio_path "${AUDIO_PATH}" \
  --prompt "A person speaks clearly in a quiet room, natural lighting, cinematic medium shot." \
  --negative_prompt "blurry, out of focus, overexposed, underexposed, low contrast, excessive noise, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, disfigured hands, artifacts, inconsistent perspective, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, off-sync audio, AI artifacts." \
  --save_result_path "${lightx2v_path}/save_results/output_lightx2v_ltx2_s2v_metax.mp4" \
