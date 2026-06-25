#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme5/gushiqiao/codes/LightX2V
model_path=/data/nvme5/gushiqiao/models/Cosmos3-Super-Text2Image

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls cosmos3 \
--task t2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/cosmos3/cosmos3_super_t2i.json \
--prompt "一只小猫" \
--negative_prompt "" \
--save_result_path ${lightx2v_path}/save_results/cosmos3_t2i.png \
--seed 1143
