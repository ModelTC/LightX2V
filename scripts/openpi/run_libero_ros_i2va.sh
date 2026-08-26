#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lightx2v_path="${LIGHTX2V_PATH:-$(cd -- "${script_dir}/../.." && pwd)}"
ros_workspace="${lightx2v_path}/lightx2v_ros"
workspace_root="$(dirname -- "${lightx2v_path}")"
openpi_data_root="${OPENPI_DATA_ROOT:-${workspace_root}/openpi_data}"
openpi_path="${OPENPI_PATH:-${workspace_root}/openpi}"
model_path="${OPENPI_MODEL_PATH:-${openpi_data_root}/openpi-assets/checkpoints/pi05_libero_pytorch}"
config_json="${OPENPI_CONFIG:-${lightx2v_path}/configs/openpi/pi05_libero.json}"
libero_root="${OPENPI_LIBERO_ROOT:-${openpi_path}/third_party/libero}"
transformers_runtime_path="${OPENPI_TRANSFORMERS_RUNTIME_PATH:-${openpi_data_root}/python_deps/openpi_pytorch_runtime}"
ros_setup="${ROS_SETUP:-${HOME:-}/ros2_lyrical/install/setup.sh}"
if [[ -z "${ROS_SETUP:-}" && ! -f "${ros_setup}" ]]; then
    ros_setup="/opt/ros/jazzy/setup.bash"
fi

if [[ ! -f "${ros_setup}" ]]; then
    echo "ROS setup not found: ${ros_setup}. Set ROS_SETUP to your ROS2 setup script." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-}"

set +u
source "${ros_setup}"
set -u
source "${lightx2v_path}/scripts/base/base.sh"

cd "${ros_workspace}"
colcon build --symlink-install --packages-select common simulator inference
set +u
source "${ros_workspace}/install/setup.bash"
set -u

simulator_pid=""
cleanup() {
    if [[ -n "${simulator_pid}" ]]; then
        kill "${simulator_pid}" 2>/dev/null || true
        wait "${simulator_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

ros2 run simulator libero_node --ros-args \
    -p autostart:=true \
    -p "libero_root:=${libero_root}" \
    -p "benchmark:=${LIBERO_BENCHMARK:-libero_10}" \
    -p "task_id:=${LIBERO_TASK_ID:-5}" \
    -p "init_state_id:=${LIBERO_INIT_STATE_ID:-0}" \
    -p "seed:=${LIBERO_SEED:-0}" &
simulator_pid=$!

PYTHONPATH="${transformers_runtime_path}:${PYTHONPATH}" \
USE_FLAX=0 \
ros2 run inference openpi_node --ros-args \
    -p env:=libero \
    -p "model_path:=${model_path}" \
    -p "config_json:=${config_json}" \
    -p "seed:=${OPENPI_SEED:-7}" \
    -p "actions_per_plan:=${OPENPI_ACTIONS_PER_PLAN:-5}" \
    -p "num_steps_wait:=${OPENPI_NUM_STEPS_WAIT:-10}"
