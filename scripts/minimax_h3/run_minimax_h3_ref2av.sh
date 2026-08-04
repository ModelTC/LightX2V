#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
lightx2v_path=${LIGHTX2V_PATH:-${REPO_ROOT}}
model_path=${MODEL_PATH:-/data/nvme6/gushiqiao/models/MiniMax-H3}
export PYTHONPATH=${PYTHONPATH:-}
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32
OUTPUT_PATH=${OUTPUT_PATH:-${lightx2v_path}/outputs/minimax_h3_ref2av.mp4}
mkdir -p "$(dirname -- "${OUTPUT_PATH}")"
reference_args=()
IFS=',' read -ra refs <<< "${REFERENCES:?set REFERENCES to comma-separated kind=/path entries}"
for ref in "${refs[@]}"; do reference_args+=(--reference "${ref}"); done
python -m lightx2v.infer --model_cls minimax_h3 --task ref2av \
  --model_path "${model_path}" \
  --config_json "${CONFIG_JSON:-${lightx2v_path}/configs/minimax_h3/minimax_h3_ref2av.json}" \
  "${reference_args[@]}" --prompt "${PROMPT:-Generate an audio-video scene following the references.}" \
  --seed "${SEED:-42}" --save_result_path "${OUTPUT_PATH}"
