#!/bin/bash
set -euo pipefail

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${MODEL_PATH:-${lightx2v_path}/weights/hymotion}"
config_json="${CONFIG_JSON:-${lightx2v_path}/configs/platforms/mlu/hymotion_t2m_bf16.json}"
save_result_path="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/hymotion/result.npz}"

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: MODEL_PATH=/local/weights PYTHON_BIN=python [PROMPT=...] bash scripts/platforms/mlu/run_hymotion.sh [lightx2v.infer arguments]"
  echo "Preparation: scripts/hymotion/README.md; single-device torch.mlu inference"
  exit 0
fi

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${lightx2v_path}:${PYTHONPATH:-}"
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
mkdir -p -- "$(dirname -- "${save_result_path}")"
cd -- "${lightx2v_path}"

"${PYTHON_BIN:-python}" -c 'import torch_mlu; from lightx2v.infer import main; main()' \
  --model_cls hymotion \
  --task t2m \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --prompt "${PROMPT:-A person walks forward at a steady pace.}" \
  --save_result_path "${save_result_path}" \
  --seed "${SEED:-42}" \
  "$@"
