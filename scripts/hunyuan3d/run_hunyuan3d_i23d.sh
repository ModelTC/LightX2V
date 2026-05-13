#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/wushuo/lyra_proj/LightX2V
model_path=/data/nvme1/wushuo/hf_models/Hunyuan3D-2.1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

image_path=/data/nvme1/wushuo/lyra_proj/Hunyuan3D-2.1/assets/demo.png
save_result_path=${lightx2v_path}/save_results/hunyuan3d_i23d

python -m lightx2v.infer \
    --model_cls hunyuan3d \
    --model_path ${model_path} \
    --config_json ${lightx2v_path}/configs/hunyuan3d/hunyuan3d_i23d.json \
    --task i23d \
    --image_path ${image_path} \
    --save_result_path ${save_result_path} \
    --seed 42
