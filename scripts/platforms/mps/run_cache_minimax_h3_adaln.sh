#!/bin/bash
set -euo pipefail

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to the MiniMax-H3 diffusers checkpoint directory}"
model_path="${MODEL_PATH}"
export PLATFORM=mps
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
export CONFIG_JSON="${CONFIG_JSON:-${lightx2v_path}/configs/platforms/mps/minimax_h3_t2av_4step_512_22.json}"
export PYTHONPATH="${PYTHONPATH:-}"
source "${lightx2v_path}/scripts/base/base.sh"

# Use the same model and JSON config as inference when generating the cache.
exec "${PYTHON:-python}" "${lightx2v_path}/tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py" \
    --model_path "${model_path}" \
    --config_json "${CONFIG_JSON}" \
    --model-variant fl2av \
    "$@"
