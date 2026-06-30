#!/bin/bash

# set path firstly
lightx2v_path=/data/LightX2V
model_path=/data/models/Wan2.2-T2V-A14B

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES=0
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:${LD_LIBRARY_PATH}

source ${lightx2v_path}/scripts/platforms/mlu/logging.sh

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

mkdir -p ${lightx2v_path}/save_results

python -m lightx2v.infer \
--model_cls wan2.2_moe \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/platforms/mlu/single/wan_moe_t2v.json \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan22_moe_t2v_mlu.mp4
