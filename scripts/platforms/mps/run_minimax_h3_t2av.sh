#!/bin/bash
set -euo pipefail

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to the MiniMax-H3 diffusers checkpoint directory}"
config_json="${CONFIG_JSON:-${lightx2v_path}/configs/platforms/mps/minimax_h3_t2av_4step_512_22.json}"
save_result_path="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av.mp4}"

export PLATFORM=mps
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${lightx2v_path}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "$(dirname -- "${save_result_path}")"

exec "${PYTHON:-python}" -m lightx2v.infer \
    --model_cls minimax_h3 \
    --model-variant fl2av \
    --task t2av \
    --model_path "${MODEL_PATH}" \
    --config_json "${config_json}" \
    --prompt "${PROMPT:-A cinematic fox walking through a snowy forest}" \
    --save_result_path "${save_result_path}" \
    --seed "${SEED:-42}" \
    "$@"
