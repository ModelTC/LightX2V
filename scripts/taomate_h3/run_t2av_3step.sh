#!/usr/bin/env bash
set -euo pipefail

lightx2v_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_path="${MODEL_PATH:-${lightx2v_path}/../models/H3/MiniMax-H3}"
config_path="${CONFIG_PATH:-${lightx2v_path}/configs/taomate_h3/t2av_10s_bf16_sage_5090_tp2_sp4.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
  --model_cls taomate_h3 \
  --model-variant fl2av \
  --task t2av \
  --model_path "${model_path}" \
  --config_json "${config_path}" \
  --prompt "Official prompts are loaded from the config prompt_json." \
  --seed 8301 \
  --save_result_path "${OUTPUT_PATH:-${lightx2v_path}/save_results/taomate_h3_10s.mp4}"
