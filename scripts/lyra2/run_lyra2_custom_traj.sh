#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/wushuo/lyra_proj/LightX2V
model_path=/data/nvme1/wushuo/hf_models/nvidia/Lyra-2.0/checkpoints

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# Per-run inputs
image_path=/data/nvme1/wushuo/lyra_proj/Lyra-2/assets/custom_trajectory_examples/example_0/first_frame.png
trajectory_path=/data/nvme1/wushuo/lyra_proj/Lyra-2/assets/custom_trajectory_examples/example_0/trajectory.npz

# prompt: plain text string OR path to a per-chunk captions .json file
# If a .json path is given, per-AR-chunk captions are used automatically.
prompt=/data/nvme1/wushuo/lyra_proj/Lyra-2/assets/custom_trajectory_examples/example_0/captions.json

save_result_path=${lightx2v_path}/save_results/lyra2_custom_traj

python -m lightx2v.infer \
    --model_cls lyra2_custom_traj \
    --model_path ${model_path} \
    --config_json ${lightx2v_path}/configs/lyra2/lyra2_custom_traj.json \
    --task lyra2_custom_traj \
    --image_path ${image_path} \
    --trajectory_path ${trajectory_path} \
    --prompt "${prompt}" \
    --save_result_path ${save_result_path} \
    --seed 1
