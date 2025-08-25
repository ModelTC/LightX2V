#!/bin/bash

# set path and first
lightx2v_path=
model_path=/data/nvme0/models/Matrix-Game-2.0
model_ckpt_path=/data/nvme0/models/Matrix-Game-2.0/base_distilled_model/base_distill.safetensors

export CUDA_VISIBLE_DEVICES=0

# set environment variables
export matrix_game_config_path=${lightx2v_path}/configs/matrixgame/model_config.json
export model_ckpt_path=${model_ckpt_path}
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls wan2.1_matrixgame \
--task i2v \
--image_path ${lightx2v_path}/assets/inputs/imgs/img_0.jpg \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/matrixgame/wan_i2v_matrixgame.json \
--save_video_path ${lightx2v_path}/save_results/output_lightx2v_wan_i2v_causvid.mp4
