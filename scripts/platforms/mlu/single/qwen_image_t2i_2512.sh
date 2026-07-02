#!/bin/bash

# System management interface: cnmon

# set path firstly
lightx2v_path=/data/LightX2V
model_path=/data/models/Qwen-Image-2512

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES=0
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:${LD_LIBRARY_PATH}

# set environment variables
source "${lightx2v_path}/scripts/platforms/mlu/single/logging.sh"
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
--model_cls qwen_image \
--task t2i \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/platforms/mlu/single/qwen_image_t2i_2512.json \
--prompt 'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition, Ultra HD, 4K, cinematic composition.' \
--negative_prompt " " \
--save_result_path ${lightx2v_path}/save_results/qwen_image_t2i_2512_mlu.png \
--seed 42
