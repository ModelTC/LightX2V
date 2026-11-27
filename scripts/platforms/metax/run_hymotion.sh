#!/bin/bash
set -euo pipefail

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${MODEL_PATH:-${lightx2v_path}/weights/hymotion}"
config_json="${CONFIG_JSON:-${lightx2v_path}/configs/platforms/metax/hymotion_t2m_bf16.json}"
save_result_path="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/hymotion/result.npz}"

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: MODEL_PATH=/local/weights PYTHON_BIN=python [PROMPT=...] bash scripts/platforms/metax/run_hymotion.sh [lightx2v.infer arguments]"
  echo "Preparation: scripts/hymotion/README.md; status: placeholder, exits without inference"
  exit 0
fi

export PLATFORM=metax_cuda
echo "This platform is a placeholder; A100 is the only implemented target." >&2
exit 2
