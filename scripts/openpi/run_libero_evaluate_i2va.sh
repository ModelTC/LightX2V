#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-$(cd -- "${script_dir}/../.." && pwd)}"
workspace_root="$(dirname -- "${lightx2v_path}")"
openpi_data_root="${OPENPI_DATA_ROOT:-${workspace_root}/openpi_data}"

model_path="${OPENPI_MODEL_PATH:-${openpi_data_root}/openpi-assets/checkpoints/pi05_libero_pytorch_fp32}"
config_json="${OPENPI_CONFIG:-${lightx2v_path}/configs/openpi/pi05_libero.json}"
output_dir="${OPENPI_EVAL_OUTPUT_DIR:-${lightx2v_path}/save_results/pi05_libero_pytorch_fp32_evaluation}"
runtime_dir="${OPENPI_RUNTIME_DIR:-${output_dir}/runtime}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OPENPI_RUN_MODE=evaluate
export OPENPI_LIBERO_ROOT="${OPENPI_LIBERO_ROOT:-${workspace_root}/openpi/third_party/libero}"
export OPENPI_LIBERO_CONFIG_DIR="${OPENPI_LIBERO_CONFIG_DIR:-${runtime_dir}/libero_config}"
export OPENPI_EVAL_CONFIG="${OPENPI_EVAL_CONFIG:-${lightx2v_path}/configs/openpi/pi05_libero_eval.json}"
export OPENPI_TRANSFORMERS_RUNTIME_PATH="${OPENPI_TRANSFORMERS_RUNTIME_PATH:-${openpi_data_root}/python_deps/openpi_official_pytorch_runtime}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${runtime_dir}/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${runtime_dir}/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${runtime_dir}/cache}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PROFILING_DEBUG_LEVEL="${PROFILING_DEBUG_LEVEL:-0}"

mkdir -p "${OPENPI_LIBERO_CONFIG_DIR}" "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"
cd "${lightx2v_path}"

exec python -m lightx2v.infer \
  --model_cls openpi \
  --task i2va \
  --model_path "${model_path}" \
  --config_json "${config_json}" \
  --seed "${OPENPI_POLICY_SEED:-0}" \
  --save_result_path "${output_dir}"
