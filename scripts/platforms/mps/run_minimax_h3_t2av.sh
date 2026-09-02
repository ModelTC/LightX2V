#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

lightx2v_path=${LIGHTX2V_PATH:-${REPO_ROOT}}
model_path=${MODEL_PATH:-}
config_json=${CONFIG_JSON:-${lightx2v_path}/configs/platforms/mps/minimax_h3_t2av.json}
output_path=${OUTPUT_PATH:-${lightx2v_path}/save_results/output_lightx2v_minimax_h3_t2av.mp4}

export PLATFORM=mps
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=BF16
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"

if [[ -z "${model_path}" ]]; then
    echo "MODEL_PATH must point to the MiniMax-H3 model directory."
    exit 1
fi

[[ -d "${model_path}" ]] || {
    echo "Model directory not found: ${model_path}"
    exit 1
}

[[ -f "${config_json}" ]] || {
    echo "Config file not found: ${config_json}"
    exit 1
}

mkdir -p "$(dirname -- "${output_path}")"

prompt=${PROMPT:-A cinematic fox walking through a snowy forest}
seed=${SEED:-42}

echo "Starting MiniMax-H3 t2av on platform=${PLATFORM}, dtype=${DTYPE}"
echo "Config: dit_disk_streaming=true, text_encoder_disk_streaming=true, VAE lazy lifecycle active"

python -m lightx2v.infer \
  --model_cls minimax_h3 \
  --task t2av \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --prompt "${prompt}" \
  --save_result_path "${output_path}" \
  --seed "${seed}"
