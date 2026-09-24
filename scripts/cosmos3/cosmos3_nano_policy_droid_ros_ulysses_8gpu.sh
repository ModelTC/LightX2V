#!/bin/bash

set -e

# Set paths.
lightx2v_path=path/to/LightX2V
model_path=path/to/Cosmos3-Nano-Policy-DROID

# Set environment variables.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /opt/ros/jazzy/setup.bash
source "${lightx2v_path}/scripts/base/base.sh"

# Build and load the ROS workspace.
cd "${lightx2v_path}/lightx2v_ros"
colcon build --symlink-install
source "install/setup.bash"

cd "${lightx2v_path}"
exec torchrun --nproc_per_node=8 -m inference.cosmos3_node.main \
  --ros-args \
  -p env:=robolab \
  -p "model_path:=${model_path}" \
  -p "config_json:=${lightx2v_path}/configs/cosmos3/cosmos3_nano_policy_droid_cfg_ulysses_8gpu.json" \
  -p actions_per_plan:=32 \
  -p binarize_gripper:=true \
  -p prompt_format:=official_text \
  -p seed:=0
