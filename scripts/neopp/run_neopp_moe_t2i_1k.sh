#!/bin/bash

# set path firstly
lightx2v_path=/data/nvme1/yongyang/FL/LightX2V
model_path=/data/nvme1/yongyang/FL/neo_gen_30b_moe/neo_gen_30b_moe

export CUDA_VISIBLE_DEVICES=5

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--seed 200 \
--model_cls neopp \
--task t2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/neopp/neopp_moe_t2i.json \
--prompt "a photo of two trucks" \
--target_shape 1024 1024 \
--negative_prompt "" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_neopp_moe_t2i_1k.png
