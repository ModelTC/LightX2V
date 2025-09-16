#!/bin/bash

# set path and first
lightx2v_path=/mnt/afs/users/lijiaqi2/deploy-comfyui-ljq-custom_nodes/ComfyUI-Lightx2vWrapper/lightx2v
model_path=/mnt/afs/users/lijiaqi2/wan_model/Wan2.1-R2V0909-Audio-14B-720P-fp8


export CUDA_VISIBLE_DEVICES=0,1,2,3

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

# Start multiple servers
torchrun --nproc_per_node 4 -m lightx2v.server \
    --model_cls seko_talk \
    --task i2v \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/seko_talk/xxx_dist.json \
    --port 8000
    
    