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
OUTPUT_PATH=${OUTPUT_PATH:-${lightx2v_path}/outputs/minimax_h3_l2av.mp4}
mkdir -p "$(dirname -- "${OUTPUT_PATH}")"
python -m lightx2v.infer --model_cls minimax_h3 --task l2av \
  --model_path "${model_path}" \
  --config_json "${CONFIG_JSON:-${lightx2v_path}/configs/minimax_h3/minimax_h3_l2av.json}" \
  --last_frame_path "${LAST_FRAME_PATH:?set LAST_FRAME_PATH}" --prompt "${PROMPT:-Generate the preceding scene with natural synchronized sound.}" \
  --seed "${SEED:-42}" --save_result_path "${OUTPUT_PATH}"
