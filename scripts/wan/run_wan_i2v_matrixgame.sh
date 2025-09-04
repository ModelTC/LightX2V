#!/bin/bash

# set path and first
lightx2v_path=
model_path=/data/nvme0/models/Matrix-Game-2.0

export CUDA_VISIBLE_DEVICES=0

# set environment variables
export matrix_game_config_path=${lightx2v_path}/configs/matrixgame/universal_config.json
# export matrix_game_config_path=${lightx2v_path}/configs/matrixgame/gta_drive_config.json
# export matrix_game_config_path=${lightx2v_path}/configs/matrixgame/templerun_config.json
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls wan2.1_matrixgame \
--task i2v \
--image_path ${lightx2v_path}/assets/inputs/imgs/matrixgame_input_universal_0.png \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/matrixgame/wan_i2v_matrixgame.json \
--save_video_path ${lightx2v_path}/save_results/output_lightx2v_wan_i2v_matrixgame.mp4
