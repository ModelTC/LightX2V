#!/usr/bin/env bash

set -euo pipefail

lightx2v_path=/path/to/LightX2V
ros_workspace=${lightx2v_path}/lightx2v_ros
model_path=/path/to/lingbot-va-posttrain-robotwin
ros_setup=/path/to/ros2/setup.bash

export CUDA_VISIBLE_DEVICES=0

set +u
source "${ros_setup}"
source ${lightx2v_path}/scripts/base/base.sh
set -u

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

ros2 run simulator robotwin_node --ros-args \
    -p autostart:=true \
    -p task_name:=hanging_mug \
    -p task_config:=demo_clean \
    -p seed:=0 &
simulator_pid=$!

ros2 run inference lingbot_va_node --ros-args \
    -p env:=robotwin \
    -p "model_path:=${model_path}" \
    -p config_json:=${lightx2v_path}/configs/lingbot_va/robotwin_i2va.json \
    -p seed:=0 \
    -p num_steps_wait:=0
