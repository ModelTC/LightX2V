#!/bin/bash

set -eo pipefail

lightx2v_path=/data/LightX2V
model_path=/data/models/LTX-2
audio_path="${lightx2v_path}/assets/inputs/audio/seko_input.mp3"

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_MLU_ALLOC_CONF="${PYTORCH_MLU_ALLOC_CONF:-expandable_segments:True}"
export LD_LIBRARY_PATH="/usr/local/neuware/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONFAULTHANDLER=1

source "${lightx2v_path}/scripts/base/base.sh"

log_dir="${lightx2v_path}/save_results/logs/ltx2_3_s2v_mlu_8card"
mkdir -p "${log_dir}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  --log-dir="${log_dir}" \
  --redirects=3 \
  --tee=3 \
  -m lightx2v.infer \
  --model_cls ltx2 \
  --task ltx2_s2v \
  --model_path "${model_path}" \
  --config_json "${lightx2v_path}/configs/platforms/mlu/ltx2_3_dist.json" \
  --audio_path "${audio_path}" \
  --target_video_length 121 \
  --prompt "A person speaks clearly in a quiet room, natural lighting, cinematic medium shot." \
  --negative_prompt "blurry, out of focus, overexposed, underexposed, low contrast, excessive noise, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, disfigured hands, artifacts, inconsistent perspective, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, off-sync audio, AI artifacts." \
  --save_result_path "${lightx2v_path}/save_results/output_lightx2v_ltx2_3_s2v_mlu_8card.mp4" \
  --seed 42
