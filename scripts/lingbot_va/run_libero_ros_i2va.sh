#!/usr/bin/env bash

set -euo pipefail

lightx2v_path="${LIGHTX2V_PATH:-/data/nvme7/yongyang/LightX2V}"
ros_workspace="${lightx2v_path}/lightx2v_ros"
model_path="${LINGBOT_VA_LIBERO_MODEL_PATH:-/data/nvme5/gushiqiao/models/lingbot-va-posttrain-libero-long}"
config_json="${LINGBOT_VA_LIBERO_CONFIG:-${lightx2v_path}/configs/lingbot_va/libero_i2va.json}"
lyrical_setup="${HOME:-}/ros2_lyrical/install/setup.sh"
if [[ -n "${ROS_SETUP:-}" ]]; then
    ros_setup="${ROS_SETUP}"
elif [[ -f "${lyrical_setup}" ]]; then
    ros_setup="${lyrical_setup}"
else
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
    -p "benchmark:=${LIBERO_BENCHMARK:-libero_10}" \
    -p "task_id:=${LIBERO_TASK_ID:-5}" \
    -p "init_state_id:=${LIBERO_INIT_STATE_ID:-0}" \
    -p "seed:=${LIBERO_SEED:-0}" &
simulator_pid=$!

ros2 run inference lingbot_va_node --ros-args \
    -p env:=libero \
    -p "model_path:=${model_path}" \
    -p "config_json:=${config_json}" \
    -p "seed:=${LINGBOT_VA_SEED:-0}" \
    -p "num_steps_wait:=${LINGBOT_VA_NUM_STEPS_WAIT:-5}" \
    -p "undo_libero_horizontal_flip:=${LINGBOT_VA_UNDO_LIBERO_HORIZONTAL_FLIP:-true}"
