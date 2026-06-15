#!/bin/bash

# set path firstly
lightx2v_path=/data/wushuo1/LightX2V
model_path=/data/wushuo1/to5_models/Qwen-Image-2512

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls qwen_image \
--task t2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/platforms/ascend_npu/qwen_image_t2i_2512.json \
--prompt 'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition, Ultra HD, 4K, cinematic composition.' \
--negative_prompt " " \
--save_result_path ${lightx2v_path}/save_results/qwen_image_t2i_2512.png \
--seed 42
