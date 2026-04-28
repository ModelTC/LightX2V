#!/bin/bash

# set path firstly
lightx2v_path=/path/to/LightX2V
model_path=/path/to/MotusModel

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls motus \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/motus/motus_i2v.json \
--image_path "/path/to/the/first/frame: example_inputs/frist_frame.png" \
--state_path "/path/to/the/state/at/the/first/frame: example_inputs/state.npy" \
--prompt "Example prompt: The whole scene is in a realistic, industrial art style with three views: a fixed rear camera, a movable left arm camera, and a movable right arm camera. The aloha robot is currently performing the following task: Pick the bottle with ridges near base head-up using the right arm" \
--save_result_path ${lightx2v_path}/save_results/output_motus.mp4 \
--save_action_path ${lightx2v_path}/save_results/output_motus.actions.json \
--seed 42
