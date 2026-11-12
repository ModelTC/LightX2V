#!/bin/bash
set -euo pipefail

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${MODEL_PATH:-${lightx2v_path}/weights/trellis2}"
input_path="${INPUT_PATH:-}"
config_json="${CONFIG_JSON:-${lightx2v_path}/configs/platform/ascend_npu/trellis2_i23d_bf16.json}"
save_result_path="${SAVE_RESULT_PATH:-${lightx2v_path}/../outputs/trellis2/mesh.glb}"

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: MODEL_PATH=/path/to/weights INPUT_PATH=/path/to/input bash scripts/platform/ascend_npu/run_trellis2.sh [lightx2v.infer arguments]"
  echo "Prepare local weights first: ${lightx2v_path}/scripts/trellis2/README.md"
  exit 0
fi

if [[ -z "${input_path}" ]]; then
  echo "Error: set INPUT_PATH to the input image." >&2
  exit 2
fi

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PYTHONPATH:-}"
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
mkdir -p -- "$(dirname -- "${save_result_path}")"
cd -- "${lightx2v_path}"

python -m lightx2v.infer \
  --model_cls trellis2 \
  --task i23d \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --image_path "${input_path}" \
  --save_result_path "${save_result_path}" \
  --seed "${SEED:-42}" \
  "$@"
