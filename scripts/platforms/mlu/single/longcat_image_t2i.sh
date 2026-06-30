#!/bin/bash

# LongCat Image T2I Inference Script for Cambricon MLU
# Usage: bash longcat_image_t2i.sh

# set path firstly
lightx2v_path=/data/LightX2V
model_path=/data/models/LongCat-Image

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES=0
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:${LD_LIBRARY_PATH}

source ${lightx2v_path}/scripts/platforms/mlu/logging.sh

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

mkdir -p ${lightx2v_path}/save_results

python -m lightx2v.infer \
    --model_cls longcat_image \
    --task t2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/platforms/mlu/single/longcat_image_t2i.json \
    --prompt "一只小猫躺在沙发上" \
    --negative_prompt "" \
    --save_result_path ${lightx2v_path}/save_results/longcat_image_t2i_mlu.png \
    --seed 42
