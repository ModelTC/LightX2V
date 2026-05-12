#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/wushuo/lyra_proj/LightX2V
model_path=/data/nvme1/wushuo/hf_models/nvidia/Lyra-2.0/checkpoints

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

image_path=/data/nvme1/wushuo/lyra_proj/Lyra-2/assets/samples/01.png
prompt="A cinematic forward camera glide along the stone balcony of a red wooden pagoda. The scene is frozen in time; every element is perfectly still. To the left, the intricate vermilion architecture, tiered eaves, and textured stone floor remain motionless. To the right, the vast sea of white cumulus clouds and the soft blue sky are static, with no shifting vapor or light. The shadows across the walkway are fixed. This steady push-in reveals more of the balcony and distant pagoda structures, maintaining the original lighting, vibrant red palette, and serene atmosphere. All clouds, wood, and stone are entirely stationary."
save_result_path=${lightx2v_path}/save_results/lyra2_zoomgs_x2v

python -m lightx2v.infer \
    --model_cls lyra2_zoomgs \
    --model_path ${model_path} \
    --config_json ${lightx2v_path}/configs/lyra2/lyra2_zoomgs.json \
    --task i2v \
    --image_path ${image_path} \
    --prompt "${prompt}" \
    --save_result_path ${save_result_path} \
    --seed 1
