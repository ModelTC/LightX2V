#!/bin/bash
set -euo pipefail

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${MODEL_PATH:-${lightx2v_path}/weights/sdxl}"
config_json="${CONFIG_JSON:-${lightx2v_path}/configs/platforms/ascend_npu/sdxl_t2i_fp16.json}"
save_result_path="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/sdxl/result.png}"

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: MODEL_PATH=/local/weights PYTHON_BIN=python [PROMPT=...] bash scripts/platforms/ascend_npu/run_sdxl.sh [lightx2v.infer arguments]"
  echo "Preparation: scripts/sdxl/README.md; status: placeholder, exits without inference"
  exit 0
fi

export PLATFORM=ascend_npu
echo "This platform is a placeholder; A100 is the only implemented target." >&2
exit 2
