#!/bin/bash
set -euo pipefail

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${MODEL_PATH:-${lightx2v_path}/weights/sf3d}"
config_json="${CONFIG_JSON:-${lightx2v_path}/configs/platforms/ascend_npu/sf3d_i23d_fp32.json}"
save_result_path="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/sf3d/result.glb}"

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: MODEL_PATH=/local/weights PYTHON_BIN=python [PROMPT=... or INPUT_PATH=one-image] bash scripts/platforms/ascend_npu/run_sf3d.sh [lightx2v.infer arguments]"
  echo "Preparation: scripts/sf3d/README.md; status: placeholder, exits without inference"
  exit 0
fi

export PLATFORM=ascend_npu
echo "This platform is a placeholder; A100 is the only implemented target." >&2
exit 2
export PYTHONPATH="${PYTHONPATH:-}"
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
mkdir -p -- "$(dirname -- "${save_result_path}")"
cd -- "${lightx2v_path}"

"${PYTHON_BIN:-python}" -m lightx2v.infer \
  --model_cls sf3d \
  --task i23d \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --image_path "${INPUT_PATH:?Set INPUT_PATH to one image}" \
  --save_result_path "${save_result_path}" \
  --seed "${SEED:-42}" \
  "$@"
