#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

lightx2v_path=${LIGHTX2V_PATH:-${REPO_ROOT}}
model_path=${MODEL_PATH:-/llm/models/Wan2.2-TI2V-5B}
config_json=${CONFIG_JSON:-${lightx2v_path}/configs/platforms/intel_xpu/dist_infer/wan22_ti2v_t2v_tp.json}
output_path=${OUTPUT_PATH:-${lightx2v_path}/save_results/output_wan22_ti2v_t2v_tp.mp4}

export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export PYTHONPATH=${PYTHONPATH:-}

source "${lightx2v_path}/scripts/base/base.sh"
mkdir -p "$(dirname -- "${output_path}")"

torchrun --standalone --nproc_per_node=2 -m lightx2v.infer \
  --model_cls wan2.2 \
  --task t2v \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --prompt "${PROMPT:-Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage}" \
  --negative_prompt "${NEGATIVE_PROMPT:-色调艳丽，过曝，静态，细节模糊不清，字幕，低质量，JPEG压缩残留，畸形，多余的手指，杂乱的背景}" \
  --seed "${SEED:-42}" \
  --save_result_path "${output_path}"
