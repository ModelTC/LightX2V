#!/bin/bash

lightx2v_path=/path/to/LightX2V
model_path=/path/to/LongCat-Image

export CUDA_VISIBLE_DEVICES=2,3

source ${lightx2v_path}/scripts/base/base.sh
torchrun --nproc_per_node=2 -m lightx2v.infer \
--model_cls longcat_image \
--task t2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/longcat_image/longcat_image_t2i_cfg_parallel.json \
--prompt "一只小猫躺在沙发上" \
--negative_prompt "" \
--save_result_path ${lightx2v_path}/save_results/longcat_image_t2i_cfg_parallel.png \
--seed 42
