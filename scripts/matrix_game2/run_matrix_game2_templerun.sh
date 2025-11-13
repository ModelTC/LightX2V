#!/bin/bash

# set path and first
lightx2v_path=path to Lightx2v
model_path=path to Skywork/Matrix-Game-2.0

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls wan2.1_sf_mtxg2 \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/matrix_game2/matrix_game2_templerun.json \
--prompt '' \
--image_path templerun/0005.png \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_matrix_game2_templerun.mp4 \
--seed 42
