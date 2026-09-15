#!/bin/bash

# set path and first
lightx2v_path=/data/nvme1/zhangbilang/LightX2V
model_path=/data/nvme0/models/LTX-2


export CUDA_VISIBLE_DEVICES=1

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls ltx2 \
--task t2av \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/ltx2/ltx2_3_upsample_compile.json \
--prompt "A beautiful sunset over the ocean" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_ltx2_3_t2av_8_upsample.mp4
