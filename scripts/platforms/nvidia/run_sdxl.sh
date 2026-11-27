#!/bin/bash
set -euo pipefail

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${MODEL_PATH:-${lightx2v_path}/weights/sdxl}"
config_json="${CONFIG_JSON:-${lightx2v_path}/configs/platforms/nvidia/sdxl_t2i_bf16.json}"
save_result_path="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/sdxl/result.png}"

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: MODEL_PATH=/local/weights PYTHON_BIN=python [PROMPT=...] bash scripts/platforms/nvidia/run_sdxl.sh [lightx2v.infer arguments]"
  echo "Preparation: scripts/sdxl/README.md; status: implemented on A100"
  exit 0
fi

export PLATFORM=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-}"
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
mkdir -p -- "$(dirname -- "${save_result_path}")"
cd -- "${lightx2v_path}"

"${PYTHON_BIN:-python}" -m lightx2v.infer \
  --model_cls sdxl \
  --task t2i \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --prompt "${PROMPT:-A red apple on a wooden table, studio photograph.}" \
  --save_result_path "${save_result_path}" \
  --seed "${SEED:-42}" \
  "$@"
