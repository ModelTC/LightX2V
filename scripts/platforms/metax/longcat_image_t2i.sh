#!/bin/bash

# LongCat Image T2I Inference Script for MetaX
# System management interface: mx-smi

lightx2v_path=/data/LightX2V
model_path=/data/models/LongCat-Image

export PLATFORM="metax_cuda"
export CUDA_VISIBLE_DEVICES=0
export MACA_PATH=${MACA_PATH:-/opt/maca-3.3.0}
export PATH=${MACA_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${LD_LIBRARY_PATH}


source ${lightx2v_path}/scripts/base/base.sh

mkdir -p ${lightx2v_path}/save_results

python -m lightx2v.infer \
    --model_cls longcat_image \
    --task t2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/platforms/metax/longcat_image_t2i.json \
    --prompt "一只小猫躺在沙发上" \
    --negative_prompt "" \
    --save_result_path ${lightx2v_path}/save_results/longcat_image_t2i_metax.png \
    --seed 42
