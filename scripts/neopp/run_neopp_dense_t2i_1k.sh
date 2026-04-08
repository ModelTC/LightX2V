#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/yongyang/FL/LightX2V
model_path=/data/nvme1/yongyang/FL/neo_9b_new/hf_step4000_ema

export CUDA_VISIBLE_DEVICES=3

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--seed 200 \
--model_cls neopp \
--support_tasks t2i i2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/neopp/neopp_dense.json \
--target_shape 1024 1024 \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_neopp_dense_t2i_1k.png
