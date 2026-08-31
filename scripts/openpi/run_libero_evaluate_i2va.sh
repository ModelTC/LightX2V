#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-$(cd -- "${script_dir}/../.." && pwd)}"
workspace_root="$(dirname -- "${lightx2v_path}")"
openpi_data_root="${OPENPI_DATA_ROOT:-${workspace_root}/openpi_data}"
openpi_path="${OPENPI_PATH:-${workspace_root}/openpi}"
model_path="${OPENPI_MODEL_PATH:-${openpi_data_root}/openpi-assets/checkpoints/pi05_libero_pytorch}"
model_config="${OPENPI_CONFIG:-${lightx2v_path}/configs/openpi/pi05_libero.json}"
eval_config="${OPENPI_EVAL_CONFIG:-${lightx2v_path}/configs/openpi/pi05_libero_eval.json}"
output_dir="${OPENPI_EVAL_OUTPUT_DIR:-${lightx2v_path}/save_results/openpi_pi05_libero_evaluation}"
python_bin="${OPENPI_PYTHON:-python}"

export OPENPI_LIBERO_ROOT="${OPENPI_LIBERO_ROOT:-${openpi_path}/third_party/libero}"
export OPENPI_TRANSFORMERS_RUNTIME_PATH="${OPENPI_TRANSFORMERS_RUNTIME_PATH:-${openpi_data_root}/python_deps/openpi_pytorch_runtime}"
export OPENPI_LIBERO_CONFIG_DIR="${OPENPI_LIBERO_CONFIG_DIR:-${openpi_data_root}/runtime_configs/lightx2v_openpi_libero}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/lightx2v-openpi-numba-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lightx2v-openpi-matplotlib-cache}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

source "${lightx2v_path}/scripts/base/base.sh"

OPENPI_RUN_MODE=evaluate \
OPENPI_EVAL_CONFIG="${eval_config}" \
"${python_bin}" -m lightx2v.infer \
  --model_cls openpi \
  --task i2va \
  --model_path "${model_path}" \
  --config_json "${model_config}" \
  --save_result_path "${output_dir}"
