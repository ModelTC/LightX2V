#!/bin/bash

lightx2v_path=/data/nvme5/gushiqiao/codes/new/LightX2V
model_path=/data/nvme0/gushiqiao/models/official_models/LTX-2/

export CUDA_VISIBLE_DEVICES=6

source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls ltx2_ar \
--task t2av \
--model_path ${model_path} \
--config_json ${lightx2v_path}/configs/ltx2/ltx2_3_ar.json \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--save_result_path ${lightx2v_path}/save_results/output_ltx2_3_ar_t2av.mp4
