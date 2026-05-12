#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/wushuo/lyra_proj/LightX2V
# model_path points directly to the DA3 reconstruction checkpoint file
model_path=/data/nvme1/wushuo/hf_models/nvidia/Lyra-2.0/checkpoints/recon/model.pt

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# Per-run inputs
video_path=/data/nvme1/wushuo/lyra_proj/LightX2V/save_results/lyra2_custom_traj/first_frame.mp4
save_result_path=${lightx2v_path}/save_results/lyra2_gs_recon

python -m lightx2v.infer \
    --model_cls lyra2_gs_recon \
    --model_path ${model_path} \
    --config_json ${lightx2v_path}/configs/lyra2/lyra2_gs_recon.json \
    --task lyra2_gs_recon \
    --video_path ${video_path} \
    --save_result_path ${save_result_path} \
    --seed 1
