#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-$(cd -- "${script_dir}/../.." && pwd)}"
workspace_root="$(dirname -- "${lightx2v_path}")"
openpi_data_root="${OPENPI_DATA_ROOT:-${workspace_root}/openpi_data}"
ros_setup="${ROS_SETUP:-${workspace_root}/ros2_jazzy/install/setup.bash}"
ros_overlay="${LIGHTX2V_ROS_SETUP:-${lightx2v_path}/lightx2v_ros/install/local_setup.bash}"
mode="${1:-}"
domain_id="${ROS_DOMAIN_ID:-0}"

if [[ "${mode}" != "simulator" && "${mode}" != "policy" && "${mode}" != "evaluate" ]]; then
  echo "Usage: bash scripts/openpi/run_libero_ros_i2va.sh {simulator [suite task_id init_state_id]|policy|evaluate [suite]}" >&2
  exit 2
fi
if [[ ! -f "${ros_setup}" || ! -f "${ros_overlay}" ]]; then
  echo "Build and source the ROS workspace first; missing ${ros_setup} or ${ros_overlay}" >&2
  exit 1
fi

set +u
source "${ros_setup}"
source "${ros_overlay}"
set -u

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/lightx2v-openpi-ros-${domain_id}-numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lightx2v-openpi-ros-${domain_id}-matplotlib}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/lightx2v-openpi-ros-${domain_id}-log}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/tmp/lightx2v-openpi-ros-${domain_id}-libero-config}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}" "${ROS_LOG_DIR}" "${LIBERO_CONFIG_PATH}"

if [[ "${mode}" == "simulator" ]]; then
  suite="${2:-libero_spatial}"
  task_id="${3:-0}"
  init_state_id="${4:-0}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
  unset MUJOCO_EGL_DEVICE_ID
  exec ros2 run simulator libero_node --ros-args \
    -p "autostart:=${OPENPI_ROS_AUTOSTART:-true}" \
    -p loop:=false \
    -p numeric_precision:=float64 \
    -p settle_steps:=10 \
    -p "libero_root:=${OPENPI_LIBERO_ROOT:-${workspace_root}/openpi/third_party/libero}" \
    -p "benchmark:=${suite}" \
    -p "task_id:=${task_id}" \
    -p "init_state_id:=${init_state_id}" \
    -p image_size:=256 \
    -p seed:=7
fi

if [[ "${mode}" == "evaluate" ]]; then
  suite="${2:-libero_spatial}"
  exec ros2 run simulator libero_evaluate --ros-args \
    -p "task_suite_name:=${suite}" \
    -p "output_dir:=${OPENPI_ROS_OUTPUT_DIR:-${lightx2v_path}/save_results/openpi_ros_evaluation}" \
    -p "command_timeout:=${OPENPI_ROS_COMMAND_TIMEOUT:-180.0}" \
    -p "overwrite:=${OPENPI_ROS_OVERWRITE:-false}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export USE_FLAX=0
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${OPENPI_TRANSFORMERS_RUNTIME_PATH:-${openpi_data_root}/python_deps/openpi_official_pytorch_runtime}:${lightx2v_path}:${PYTHONPATH:-}"

exec ros2 run inference openpi_node --ros-args \
  -p numeric_precision:=float64 \
  -p "model_path:=${OPENPI_MODEL_PATH:-${openpi_data_root}/openpi-assets/checkpoints/pi05_libero_pytorch_fp32}" \
  -p "config_json:=${OPENPI_CONFIG:-${lightx2v_path}/configs/openpi/pi05_libero.json}" \
  -p seed:=0 \
  -p actions_per_plan:=5
