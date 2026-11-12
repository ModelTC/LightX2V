#!/bin/bash
set -euo pipefail

lightx2v_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
model_path="${MODEL_PATH:-${lightx2v_path}/weights/vggt}"
input_path="${INPUT_PATH:-}"
config_json="${CONFIG_JSON:-${lightx2v_path}/configs/vggt/vggt_recon_bf16.json}"
save_result_path="${SAVE_RESULT_PATH:-${lightx2v_path}/save_results/vggt/reconstruction.npz}"

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: MODEL_PATH=/path/to/weights INPUT_PATH=/path/to/input bash scripts/vggt/run_vggt.sh [lightx2v.infer arguments]"
  echo "Prepare local weights first: ${lightx2v_path}/scripts/vggt/README.md"
  exit 0
fi

if [[ -z "${input_path}" ]]; then
  echo "Error: set INPUT_PATH to the input image or image directory." >&2
  exit 2
fi

export PLATFORM=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-}"
source "${lightx2v_path}/scripts/base/base.sh"
export DTYPE=BF16
mkdir -p -- "$(dirname -- "${save_result_path}")"
cd -- "${lightx2v_path}"

python -m lightx2v.infer \
  --model_cls vggt \
  --task recon \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --input_path "${input_path}" \
  --save_result_path "${save_result_path}" \
  --seed "${SEED:-42}" \
  "$@"
