#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/yongyang/FL/LightX2V
model_path=/data/nvme1/yongyang/FL/neo9b/neo9b

export CUDA_VISIBLE_DEVICES=5

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--seed 200 \
--model_cls neopp \
--task i2i \
--support_tasks t2i i2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/neopp/neopp_dense.json \
--target_shape 1024 1024 \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_neopp_dense_i2i_1k.png
