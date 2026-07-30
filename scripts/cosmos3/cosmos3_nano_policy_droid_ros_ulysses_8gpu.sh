#!/usr/bin/env bash

set -e

lightx2v_path="${LIGHTX2V_PATH:-/app/LightX2V}"
model_path="${COSMOS3_MODEL_PATH:-/app/nvidia_cosmos3_models/Cosmos3-Nano-Policy-DROID}"
config_json="${COSMOS3_CONFIG_JSON:-${lightx2v_path}/configs/cosmos3/cosmos3_nano_policy_droid_cfg_ulysses_8gpu.json}"
ros_ws="${lightx2v_path}/lightx2v_ros"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

source /opt/ros/jazzy/setup.bash
source "${lightx2v_path}/scripts/base/base.sh"

cd "${ros_ws}"
colcon build --symlink-install
source install/setup.bash

cd "${lightx2v_path}"
exec torchrun --nproc_per_node=8 -m inference.cosmos3_node.main \
    --ros-args \
    -p env:=robolab \
    -p "model_path:=${model_path}" \
    -p "config_json:=${config_json}" \
    -p "actions_per_plan:=${COSMOS3_ACTIONS_PER_PLAN:-32}" \
    -p "binarize_gripper:=${COSMOS3_BINARIZE_GRIPPER:-true}" \
    -p "prompt_format:=${COSMOS3_PROMPT_FORMAT:-official_text}" \
    -p "seed:=${COSMOS3_SEED:-0}"
