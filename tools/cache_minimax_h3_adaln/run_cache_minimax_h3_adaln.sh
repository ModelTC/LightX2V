#!/bin/bash
set -euo pipefail

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to the MiniMax-H3 diffusers checkpoint directory}"
model_path="${MODEL_PATH}"
config_json="${CONFIG_JSON:-${lightx2v_path}/configs/minimax_h3/minimax_h3.json}"

export PLATFORM="${PLATFORM:-cuda}"
if [[ "${PLATFORM}" == cuda ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi
export PYTHONPATH="${PYTHONPATH:-}"
source "${lightx2v_path}/scripts/base/base.sh"

# fl2av also generates the cache used by t2av/i2av/l2av inference.
# Cache generation and inference must use the same model and JSON config.
exec "${PYTHON:-python}" "${lightx2v_path}/tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py" \
    --model_path "${model_path}" \
    --config_json "${config_json}" \
    --model-variant fl2av \
    "$@"
