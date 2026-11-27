#!/bin/bash
set -euo pipefail

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
model_path="${MODEL_PATH:-${lightx2v_path}/weights/sf3d}"
config_json="${CONFIG_JSON:-${lightx2v_path}/configs/platforms/metax/sf3d_i23d_bf16.json}"
save_result_path="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/sf3d/result.glb}"

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: MODEL_PATH=/local/weights PYTHON_BIN=python [INPUT_PATH=one-image] bash scripts/platforms/metax/run_sf3d.sh [lightx2v.infer arguments]"
  echo "Preparation: scripts/sf3d/README.md; status: placeholder, exits without inference"
  exit 0
fi

export PLATFORM=metax_cuda
echo "This platform is a placeholder; A100 is the only implemented target." >&2
exit 2
