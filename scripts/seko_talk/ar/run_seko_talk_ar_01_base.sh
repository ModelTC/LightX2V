#!/usr/bin/env bash
set -euo pipefail

lightx2v_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${MODEL_PATH:?Set MODEL_PATH to SekoTalk-Distill-AR-0703}"
config_path="${CONFIG_PATH:-${lightx2v_path}/configs/seko_talk/ar/seko_talk_ar_fp8_sgl_480p_5090_tp2_sp4.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# set environment variables
source "${lightx2v_path}/scripts/base/base.sh"

torchrun --standalone --nproc-per-node 8 -m lightx2v.infer \
--model_cls seko_talk_ar \
--task rs2v \
--model_path "${model_path}" \
--config_json "${config_path}" \
--prompt "${PROMPT:-The video features a male speaking to the camera with arms spread out, a slightly furrowed brow, and a focused gaze.}" \
--image_path "${IMAGE_PATH:-${lightx2v_path}/assets/inputs/audio/seko_input.png}" \
--audio_path "${AUDIO_PATH:-${lightx2v_path}/assets/inputs/audio/seko_input.mp3}" \
--save_result_path "${OUTPUT_PATH:-${lightx2v_path}/save_results/seko_talk_ar_480p.mp4}" \
--video_duration "${VIDEO_DURATION:-15.0}" \
--seed "${SEED:-0}"
