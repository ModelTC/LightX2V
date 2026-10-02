#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-$(cd -- "${script_dir}/../.." && pwd)}"
workspace_root="$(dirname -- "${lightx2v_path}")"
openpi_data_root="${OPENPI_DATA_ROOT:-${workspace_root}/openpi_data}"

model_path="${OPENPI_MODEL_PATH:-${openpi_data_root}/openpi-assets/checkpoints/pi05_libero_pytorch_fp32}"
config_json="${OPENPI_CONFIG:-${lightx2v_path}/configs/openpi/pi05_libero.json}"
libero_root="${OPENPI_LIBERO_ROOT:-${workspace_root}/openpi/third_party/libero}"
transformers_runtime="${OPENPI_TRANSFORMERS_RUNTIME_PATH:-${openpi_data_root}/python_deps/openpi_official_pytorch_runtime}"
result_root="${OPENPI_LIBERO_RESULT_ROOT:-${lightx2v_path}/save_results/openpi_libero}"
runtime_dir="${OPENPI_RUNTIME_DIR:-${result_root}/runtime}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OPENPI_RUN_MODE=rollout
export OPENPI_LIBERO_ROOT="${libero_root}"
export OPENPI_LIBERO_CONFIG_DIR="${OPENPI_LIBERO_CONFIG_DIR:-${runtime_dir}/libero_config}"
export OPENPI_TRANSFORMERS_RUNTIME_PATH="${transformers_runtime}"
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
  --prompt "${OPENPI_TASK_DESCRIPTION:-}" \
  --save_result_path "${OPENPI_SAVE_VIDEO_PATH:-${result_root}/rollout.mp4}" \
  --save_action_path "${OPENPI_SAVE_ACTION_PATH:-${result_root}/actions.npy}"
