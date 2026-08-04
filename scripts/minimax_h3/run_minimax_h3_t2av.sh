#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

# scripts/base/base.sh follows the repository-wide lowercase variable
# convention. Keep uppercase aliases for convenient command-line overrides.
lightx2v_path=${LIGHTX2V_PATH:-${REPO_ROOT}}
model_path=${MODEL_PATH:-/data/nvme6/gushiqiao/models/MiniMax-H3}
export PYTHONPATH=${PYTHONPATH:-}
source "${lightx2v_path}/scripts/base/base.sh"

export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=FP32

MODEL_PATH=${model_path}
CONFIG_JSON=${CONFIG_JSON:-${lightx2v_path}/configs/minimax_h3/minimax_h3_t2av.json}
PROMPT=${PROMPT:-A cinematic fox walking through a snowy forest}
OUTPUT_PATH=${OUTPUT_PATH:-${lightx2v_path}/outputs/minimax_h3_t2av.mp4}
SEED=${SEED:-42}

mkdir -p "$(dirname -- "${OUTPUT_PATH}")"

python -m lightx2v.infer \
  --model_cls minimax_h3 \
  --task t2av \
  --model_path "${MODEL_PATH}" \
  --config_json "${CONFIG_JSON}" \
  --prompt "${PROMPT}" \
  --seed "${SEED}" \
  --save_result_path "${OUTPUT_PATH}"
