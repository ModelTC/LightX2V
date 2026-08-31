#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/ERNIE-Image

export CUDA_VISIBLE_DEVICES=0

source $lightx2v_path/scripts/base/base.sh

python -m lightx2v.infer \
    --model_cls ernie_image \
    --task t2i \
    --model_path $model_path \
    --config_json $lightx2v_path/configs/ernie_image/ernie_image_t2i.json \
    --prompt "一只黑白相间的中华田园犬" \
    --negative_prompt "" \
    --save_result_path $lightx2v_path/save_results/ernie_image_t2i.png \
    --seed 42
